#!/usr/bin/env python3
"""
Convert infer_widesearch response format to WideSearch evaluation format.

This script converts response files from:
    {model_name}_{instance_id}_{trial_idx}_response.jsonl (JSON format)
To:
    {model_name}_{instance_id}_{trial_idx}_response.jsonl (JSONL format compatible with WideSearch)

Usage:
    python convert_to_widesearch_format.py \
        --input_dir /path/to/input \
        --output_dir /path/to/output
"""

import argparse
import json
import os
import re
from pathlib import Path


def parse_filename(filename: str):
    """
    Parse response filename to extract model_name, instance_id, and trial_idx.

    Expected format: {model_name}_{instance_id}_{trial_idx}_response.jsonl
    Returns: (model_name, instance_id, trial_idx) or (None, None, None) if parsing fails
    """
    # Remove _response.jsonl suffix
    name = filename.replace("_response.jsonl", "")

    # Split by underscore and find the last part as trial_idx
    parts = name.split("_")
    if len(parts) < 3:
        return None, None, None

    # Last part should be trial_idx (a number)
    try:
        trial_idx = int(parts[-1])
    except ValueError:
        return None, None, None

    # Second to last part should be instance_id (like ws_en_001)
    # Instance ID format: ws_{lang}_{number}
    instance_id_match = re.search(r'(ws_\w+_\d+)_(\d+)_response\.jsonl$', filename)
    if instance_id_match:
        instance_id = instance_id_match.group(1)
        trial_idx = int(instance_id_match.group(2))
        # Everything before instance_id is model_name
        model_name = filename.replace(f"_{instance_id}_{trial_idx}_response.jsonl", "")
        return model_name, instance_id, trial_idx

    return None, None, None


def convert_file(input_path: str, output_path: str):
    """
    Convert a single response file to WideSearch format.

    Input format: JSON with keys: instance_id, response, message_history
    Output format: JSONL with the same data
    """
    # Read input file as JSON
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure required fields exist
    if "instance_id" not in data or "response" not in data:
        raise ValueError(f"Missing required fields in {input_path}")

    # WideSearch expects JSONL format (one JSON object per line)
    # Wrap in a list to match the expected format
    output_data = {
        "instance_id": data["instance_id"],
        "response": data["response"],
        "messages": data.get("message_history", None),
        "trial_idx": data.get("trial_idx", 0),
    }

    # Write as JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False)
        f.write("\n")

    print(f"Converted: {input_path} -> {output_path}")
    return data["instance_id"]


def main():
    parser = argparse.ArgumentParser(description="Convert response format for WideSearch")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing original response files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save converted files (default: input_dir/widesearch_format)",
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "widesearch_format")

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all response files
    input_files = list(Path(args.input_dir).glob("*_response.jsonl"))
    print(f"Found {len(input_files)} response files in {args.input_dir}")

    if not input_files:
        print(f"No *_response.jsonl files found in {args.input_dir}")
        return

    # Convert each file and collect metadata
    converted_ids = []
    file_metadata = []
    for input_file in input_files:
        output_file = os.path.join(args.output_dir, input_file.name)
        try:
            instance_id = convert_file(str(input_file), output_file)
            converted_ids.append(instance_id)

            # Parse filename to get metadata
            model_name, parsed_instance_id, trial_idx = parse_filename(input_file.name)
            if model_name and parsed_instance_id and trial_idx is not None:
                file_metadata.append({
                    "model_name": model_name,
                    "instance_id": parsed_instance_id,
                    "trial_idx": trial_idx,
                })
        except Exception as e:
            print(f"Error converting {input_file}: {e}")
            import traceback
            traceback.print_exc()

    # Get unique instance IDs and compute trial_num
    unique_instance_ids = sorted(set(converted_ids))
    max_trial_idx = max((m["trial_idx"] for m in file_metadata), default=0)
    trial_num = max_trial_idx + 1

    # Get model name (should be the same for all files)
    model_names = set(m["model_name"] for m in file_metadata)
    if len(model_names) == 1:
        model_name = model_names.pop()
    else:
        model_name = "SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.2"
        print(f"Warning: Multiple model names found, using default: {model_name}")

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Converted {len(converted_ids)} files")
    print(f"Unique instance IDs: {len(unique_instance_ids)}")
    print(f"Trial num: {trial_num}")
    print(f"Model name: {model_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Instance IDs: {', '.join(unique_instance_ids)}")
    print(f"{'='*60}")

    if converted_ids:
        # Generate shell script
        script_path = os.path.join(args.output_dir, "run_eval.sh")
        eval_command = (
            f"cd /mnt/mnt/public/zhangruize/MAS/repo/WideSearch && \\\n"
            f"python scripts/run_infer_and_eval_batching.py \\\n"
            f"  --stage eval \\\n"
            f"  --response_root {args.output_dir} \\\n"
            f"  --result_save_root {args.output_dir}/eval_results \\\n"
            f"  --instance_id {','.join(unique_instance_ids)} \\\n"
            f"  --model_config_name {model_name} \\\n"
            f"  --trial_num {trial_num}\n"
        )

        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Auto-generated evaluation script\n")
            f.write(f"# Generated from: {args.input_dir}\n")
            f.write(f"# Converted files: {len(converted_ids)}\n")
            f.write(f"# Unique instances: {len(unique_instance_ids)}\n")
            f.write(f"# Trial num: {trial_num}\n\n")
            f.write(eval_command)

        # Make script executable
        os.chmod(script_path, 0o755)

        print(f"\nGenerated evaluation script: {script_path}")
        print(f"Run it with: bash {script_path}\n")


if __name__ == "__main__":
    main()
