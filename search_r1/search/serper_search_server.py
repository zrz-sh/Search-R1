import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor
import argparse
import uvicorn

parser = argparse.ArgumentParser(description="Launch online search server with Serper API.")
parser.add_argument('--search_url', type=str, default="https://google.serper.dev/search",
                    help="URL for search engine (default: https://google.serper.dev/search)")
parser.add_argument('--topk', type=int, default=3,
                    help="Number of results to return per query")
parser.add_argument('--serper_api_key', type=str, default=None,
                    help="Serper API key for online search (will read from SERPER_API_KEY env var if not provided)")
args = parser.parse_args()

# Get API key from args or environment variable
serper_api_key = args.serper_api_key or os.environ.get('SERPER_API_KEY')
if not serper_api_key:
    raise ValueError("Serper API key must be provided via --serper_api_key or SERPER_API_KEY environment variable")

# --- Config ---
class OnlineSearchConfig:
    def __init__(
        self,
        search_url: str = "https://google.serper.dev/search",
        topk: int = 3,
        serper_api_key: Optional[str] = None,
    ):
        self.search_url = search_url
        self.topk = topk
        self.serper_api_key = serper_api_key


# --- Online Search Wrapper ---
class OnlineSearchEngine:
    def __init__(self, config: OnlineSearchConfig):
        self.config = config

    def _search_query(self, query: str):
        headers = {
            "X-API-KEY": self.config.serper_api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "q": query,
        }
        response = requests.post(self.config.search_url, json=payload, headers=headers)
        return response.json()

    def batch_search(self, queries: List[str]):
        results = []
        with ThreadPoolExecutor() as executor:
            for result in executor.map(self._search_query, queries):
                results.append(self._process_result(result))
        return results

    def _process_result(self, search_result: Dict):
        results = []

        # Process answer box if available
        answer_box = search_result.get('answerBox', {})
        if answer_box:
            title = answer_box.get('title', answer_box.get('snippet', 'No title.'))
            snippet = answer_box.get('answer', answer_box.get('snippet', 'No snippet available.'))
            results.append({
                'document': {"contents": f'"{title}"\n{snippet}'},
            })

        # Process knowledge graph if available
        knowledge_graph = search_result.get('knowledgeGraph', {})
        if knowledge_graph:
            title = knowledge_graph.get('title', 'No title.')
            description = knowledge_graph.get('description', 'No description available.')
            results.append({
                'document': {"contents": f'"{title}"\n{description}'},
            })

        # Process organic results
        organic_results = search_result.get('organic', [])
        for _, result in enumerate(organic_results[:self.config.topk]):
            title = result.get('title', 'No title.')
            snippet = result.get('snippet', 'No snippet available.')
            results.append({
                'document': {"contents": f'"{title}"\n{snippet}'},
            })

        # Process people also ask
        people_also_ask = search_result.get('peopleAlsoAsk', [])
        for _, result in enumerate(people_also_ask[:self.config.topk]):
            title = result.get('question', 'No title.')
            snippet = result.get('snippet', 'No snippet available.')
            results.append({
                'document': {"contents": f'"{title}"\n{snippet}'},
            })

        return results


# --- FastAPI Setup ---
app = FastAPI(title="Online Search Proxy Server (Serper API)")

class SearchRequest(BaseModel):
    queries: List[str]

# Instantiate global config + engine
config = OnlineSearchConfig(
    search_url=args.search_url,
    topk=args.topk,
    serper_api_key=serper_api_key,
)
engine = OnlineSearchEngine(config)

# --- Routes ---
@app.post("/retrieve")
def search_endpoint(request: SearchRequest):
    results = engine.batch_search(request.queries)
    return {"result": results}

## return {"result": List[List[{'document': {"id": xx, "content": "title" + \n + "content"}, 'score': xx}]]}

if __name__ == "__main__":
    # Launch the server. By default, it listens on http://0.0.0.0:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
