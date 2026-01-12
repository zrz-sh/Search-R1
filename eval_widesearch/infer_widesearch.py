#!/usr/bin/env python3
"""
Widesearch Inference Script with Transformers Backend

This script runs an agent-based inference pipeline for question answering tasks.
It uses transformers for model loading and supports parallel processing of multiple
questions using ThreadPoolExecutor (since model.generate is synchronous).

The agent logic follows Agent/Search-R1/infer.py exactly:
- Uses while True loop with stopping_criteria
- Checks EOS token to determine when to stop
- Saves message history for each turn

Usage:
    python infer_widesearch.py \
        --model_path /path/to/model \
        --dataset_path data/widesearch/widesearch.jsonl \
        --output_dir outputs/widesearch \
        --search_port 8000
"""

import argparse
import json
import os
import re
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import transformers


# ============== Configuration ==============

DEFAULT_PROMPT_TEMPLATE = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}
"""

# Qwen2.5 series EOS token IDs
QWEN_EOS_TOKEN_IDS = [151645, 151643]

MAX_NEW_TOKENS = 2048


# ============== Stopping Criteria ==============

class StopOnSequence(transformers.StoppingCriteria):
    """Custom stopping criterion that stops on specific sequences."""

    def __init__(self, target_sequences: List[str], tokenizer):
        self.target_ids = [
            tokenizer.encode(seq, add_special_tokens=False)
            for seq in target_sequences
        ]
        self.target_lengths = [len(ids) for ids in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        targets = [
            torch.as_tensor(target_id, device=input_ids.device)
            for target_id in self.target_ids
        ]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False


# ============== Search Client ==============

class SearchClient:
    """Synchronous client for search API."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8001,
        topk: int = 3,
    ):
        self.host = host
        self.port = port
        self.topk = topk
        self.base_url = f"http://{host}:{port}"
        self._session = requests.Session()
        # self._session.trust_env = False

    def search(self, query: str) -> Dict[str, Any]:
        """
        Execute a search query and return raw response.

        Args:
            query: Search query string

        Returns:
            Raw response dict with 'documents' and 'urls'
        """
        url = f"{self.base_url}/retrieve"
        payload = {
            "queries": [query],
            "topk": self.topk,
            "return_scores": False,
        }

        max_retries = 5
        for retry in range(max_retries):
            try:
                response = self._session.post(url, json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    # Return the first query's result
                    return result["result"][0]
                else:
                    print(f"[ERROR] Search API error: {response.status_code}, {response.text[:200]}")
            except Exception as e:
                print(f"[ERROR] Search request exception: {e}")

            if retry < max_retries - 1:
                import time
                time.sleep(2 * (retry + 1))

        return {"documents": [], "urls": []}


# ============== Tool Result Processing ==============

def process_tool_result(response: Dict[str, Any]) -> str:
    """
    Process tool results following MASToolWorker's consume_tool_response logic.

    Args:
        response: The response from search client with 'documents' and 'urls'

    Returns:
        Formatted text for the agent
    """
    documents = response.get("documents", [])
    urls = response.get("urls", [])

    if len(documents) > 0:
        doc_id_template = "[Doc {doc_id}]({url}):\n"
        text = "\n\n".join([
            doc_id_template.format(doc_id=str(k+1), url=url) + doc[:5000]
            for k, (doc, url) in enumerate(zip(documents, urls))
        ])
    else:
        text = "No search results are found."

    return text


def format_tool_message(query: str, tool_result: str) -> str:
    """
    Format tool message following MAS agent logic.

    Args:
        query: The search query
        tool_result: The processed tool result text

    Returns:
        Formatted tool message
    """
    return f"Search query: {query}\nResult: {tool_result}"


# ============== Helper Functions ==============

def get_query(text: str) -> Optional[str]:
    """Extract the last search query from text."""
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1].strip() if matches else None


# ============== Agent Runner ==============

class AgentRunner:
    """Runs the agent loop for a single question using transformers."""

    def __init__(
        self,
        instance_id: str,
        question: str,
        model,
        tokenizer,
        stopping_criteria,
        search_client: SearchClient,
        device: torch.device,
        curr_eos: List[int],
    ):
        self.instance_id = instance_id
        self.question = question.strip()
        if self.question and self.question[-1] != "?":
            self.question += "?"

        self.model = model
        self.tokenizer = tokenizer
        self.stopping_criteria = stopping_criteria
        self.search_client = search_client
        self.device = device
        self.curr_eos = curr_eos

        # Message history: list of (role, content) tuples
        self.message_history = []

    def run(self) -> Dict[str, Any]:
        """
        Run the agent loop until completion (following infer.py logic exactly).

        Returns:
            Dict with 'instance_id', 'response', 'final_response', 'message_history'
        """
        # Build initial prompt
        prompt_content = DEFAULT_PROMPT_TEMPLATE.format(question=self.question)

        # Apply chat template if available
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_content}],
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            prompt = prompt_content

        # Record initial user message
        self.message_history.append({
            "role": "user",
            "content": prompt_content
        })

        turn = 0
        final_response = ""

        # Main agent loop (while True, like infer.py)
        while True:
            turn += 1

            # Encode and generate
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(input_ids)

            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )

            # Check if EOS was generated (end of conversation)
            if outputs[0][-1].item() in self.curr_eos:
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                final_response = output_text

                # Record final assistant response
                self.message_history.append({
                    "role": "assistant",
                    "content": output_text
                })
                break

            # Get generated text
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            final_response = output_text

            # Extract search query from full decoded output
            full_decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            tmp_query = get_query(full_decoded)

            if tmp_query:
                # Execute search
                search_response = self.search_client.search(tmp_query)

                # Process tool result following MAS logic
                tool_result_text = process_tool_result(search_response)

                # Format tool message
                tool_message = format_tool_message(tmp_query, tool_result_text)

                # Record assistant output and tool response
                self.message_history.append({
                    "role": "assistant",
                    "content": output_text
                })
                self.message_history.append({
                    "role": "tool",
                    "content": tool_message
                })

                # Update prompt with search results (following infer.py format)
                search_text = f"\n\n{output_text}<information>{tool_result_text}</information>\n\n"
                prompt += search_text
            else:
                # No query found, record and break
                self.message_history.append({
                    "role": "assistant",
                    "content": output_text
                })
                break

        return {
            "instance_id": self.instance_id,
            "response": final_response,
            "message_history": self.message_history,
            "turns": turn,
        }


# ============== Main Inference Pipeline ==============

def run_single_question(
    question_data: Dict,
    model,
    tokenizer,
    stopping_criteria,
    search_client: SearchClient,
    device: torch.device,
    curr_eos: List[int],
    output_dir: str,
    model_name: str,
    trial_idx: int,
    lock: threading.Lock,
) -> Dict:
    """
    Process a single question.

    Args:
        question_data: Dict with 'instance_id' and 'query'
        model: The loaded model
        tokenizer: The tokenizer
        stopping_criteria: StoppingCriteriaList
        search_client: SearchClient instance
        device: torch device
        curr_eos: EOS token IDs
        output_dir: Output directory
        model_name: Model name for output files
        trial_idx: Trial index
        lock: Thread lock for model access

    Returns:
        Status dict
    """
    instance_id = question_data["instance_id"]
    query = question_data["query"]

    output_file = os.path.join(
        output_dir,
        f"{model_name}_{instance_id}_{trial_idx}_response.jsonl",
    )

    # Skip if already processed
    # if os.path.exists(output_file):
    #     print(f"[SKIP] {instance_id} already processed")
    #     return {"instance_id": instance_id, "status": "skipped"}

    try:
        runner = AgentRunner(
            instance_id=instance_id,
            question=query,
            model=model,
            tokenizer=tokenizer,
            stopping_criteria=stopping_criteria,
            search_client=search_client,
            device=device,
            curr_eos=curr_eos,
        )

        # Use lock to ensure thread-safe model access
        with lock:
            result = runner.run()

        # Save result
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "instance_id": result["instance_id"],
                    "response": result["response"],
                    "message_history": result["message_history"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"[DONE] {instance_id} completed in {result['turns']} turns")
        return {"instance_id": instance_id, "status": "success", "turns": result["turns"]}

    except Exception as e:
        import traceback
        print(f"[ERROR] {instance_id} failed: {e}")
        traceback.print_exc()
        return {"instance_id": instance_id, "status": "error", "error": str(e)}


def run_inference(
    dataset_path: str,
    output_dir: str,
    model_path: str,
    search_host: str,
    search_port: int,
    search_topk: int,
    trial_idx: int = 0,
    max_workers: int = 1,
    num_samples: int = None,
    use_timestamp: bool = True,
):
    """
    Run inference on all questions in the dataset.

    Args:
        dataset_path: Path to the JSONL dataset
        output_dir: Directory to save results
        model_path: Path to the model
        search_host: Search server host
        search_port: Search server port
        search_topk: Number of search results
        trial_idx: Trial index for output naming
        max_workers: Maximum parallel workers (usually 1 for GPU)
        num_samples: Number of samples to use (None = use all)
        use_timestamp: Whether to create timestamped subdirectory (default: True)
    """
    model_name = Path(model_path).name

    # Create timestamped subdirectory (only if use_timestamp is True)
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Load questions
    questions = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                questions.append({
                    "instance_id": data["instance_id"],
                    "query": data["query"],
                })

    # Limit number of samples if specified
    if num_samples is not None:
        questions = questions[:num_samples]

    print(f"Loaded {len(questions)} questions from {dataset_path}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Setup stopping criteria
    target_sequences = [
        "</search>", " </search>",
        "</search>\n", " </search>\n",
        "</search>\n\n", " </search>\n\n"
    ]
    stopping_criteria = transformers.StoppingCriteriaList([
        StopOnSequence(target_sequences, tokenizer)
    ])

    # Create search client
    search_client = SearchClient(
        host=search_host,
        port=search_port,
        topk=search_topk,
    )

    # Thread lock for model access
    lock = threading.Lock()

    print(f"\n{'='*50}")
    print(f"Starting inference...")
    print(f"{'='*50}\n")

    # Process questions (sequentially due to GPU constraint, but can parallelize with lock)
    results = []

    if max_workers == 1:
        # Sequential processing
        for q in questions:
            result = run_single_question(
                question_data=q,
                model=model,
                tokenizer=tokenizer,
                stopping_criteria=stopping_criteria,
                search_client=search_client,
                device=device,
                curr_eos=QWEN_EOS_TOKEN_IDS,
                output_dir=output_dir,
                model_name=model_name,
                trial_idx=trial_idx,
                lock=lock,
            )
            results.append(result)
    else:
        # Parallel processing with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_single_question,
                    q, model, tokenizer, stopping_criteria,
                    search_client, device, QWEN_EOS_TOKEN_IDS,
                    output_dir, model_name, trial_idx, lock
                ): q["instance_id"]
                for q in questions
            }

            for future in as_completed(futures):
                instance_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"[ERROR] {instance_id} exception: {e}")
                    results.append({"instance_id": instance_id, "status": "error", "error": str(e)})

    # Summary
    success_count = sum(1 for r in results if r.get("status") == "success")
    skip_count = sum(1 for r in results if r.get("status") == "skipped")
    error_count = len(results) - success_count - skip_count

    print(f"\n{'='*50}")
    print(f"Inference Complete!")
    print(f"Total: {len(results)}, Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Widesearch Inference with Transformers")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/mnt/public/zhangruize/MAS/ckpt/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.2",
        help="Path to the model",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/mnt/mnt/public/zhangruize/MAS/data/widesearch/widesearch.jsonl",
        help="Path to the dataset JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./result",
        help="Directory to save response files",
    )
    parser.add_argument(
        "--search_host",
        type=str,
        default="127.0.0.1",
        help="Search server host",
    )
    parser.add_argument(
        "--search_port",
        type=int,
        default=8001,
        help="Search server port",
    )
    parser.add_argument(
        "--search_topk",
        type=int,
        default=3,
        help="Number of search results to return",
    )
    parser.add_argument(
        "--trial_idx",
        type=int,
        default=0,
        help="Trial index for output file naming (ignored if trial_num > 1)",
    )
    parser.add_argument(
        "--trial_num",
        type=int,
        default=1,
        help="Number of trials to run (default: 1)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Maximum parallel workers (usually 1 for single GPU)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use from the dataset (default: use all samples)",
    )

    args = parser.parse_args()

    # Run multiple trials if trial_num > 1
    if args.trial_num > 1:
        # Create a shared timestamped directory for all trials
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shared_output_dir = os.path.join(args.output_dir, timestamp)
        os.makedirs(shared_output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Running {args.trial_num} trials")
        print(f"Output directory: {shared_output_dir}")
        print(f"{'='*60}\n")

        for trial_idx in range(args.trial_num):
            print(f"\n{'='*60}")
            print(f"Starting Trial {trial_idx + 1}/{args.trial_num}")
            print(f"{'='*60}\n")

            run_inference(
                dataset_path=args.dataset_path,
                output_dir=shared_output_dir,
                model_path=args.model_path,
                search_host=args.search_host,
                search_port=args.search_port,
                search_topk=args.search_topk,
                trial_idx=trial_idx,
                max_workers=args.max_workers,
                num_samples=args.num_samples,
                use_timestamp=False,  # Don't create timestamp, use shared directory
            )

            print(f"\n{'='*60}")
            print(f"Completed Trial {trial_idx + 1}/{args.trial_num}")
            print(f"{'='*60}\n")

        print(f"\n{'='*60}")
        print(f"All {args.trial_num} trials completed!")
        print(f"Results saved to: {shared_output_dir}")
        print(f"{'='*60}\n")
    else:
        run_inference(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            model_path=args.model_path,
            search_host=args.search_host,
            search_port=args.search_port,
            search_topk=args.search_topk,
            trial_idx=args.trial_idx,
            max_workers=args.max_workers,
            num_samples=args.num_samples,
            use_timestamp=True,  # Use timestamp for single trial
        )


if __name__ == "__main__":
    main()
