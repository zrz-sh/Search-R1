# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Online Search Server using Serper API

This module provides a FastAPI-based online search server that uses Serper API
for web search functionality. It exposes a /retrieve endpoint compatible with
the MASToolWorker's query_async interface.

The response format matches Agent/Search-R1/search_r1/search/online_tools.py:
- Returns List of dicts with 'documents', 'urls', and 'server_type'

Usage:
    python online_tools.py --port 8000 --topk 3

Environment Variables:
    SERPER_API_KEY: API key for Serper.dev (required)

Example:
    export SERPER_API_KEY="your-api-key"
    python online_tools.py --port 8000 --topk 3
"""

import argparse
import asyncio
import logging
import os
import random
import threading
from typing import Any, Dict, List, Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============== Logging Setup ==============

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============== Request/Response Models ==============

class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    queries: List[str]
    topk: Optional[int] = None
    return_scores: Optional[bool] = False


# ============== Configuration ==============

class OnlineSearchConfig:
    """Configuration for online search."""

    def __init__(
        self,
        topk: int = 3,
        max_retries: int = 15,
        retry_delay_base: float = 5.0,
    ):
        self.topk = topk
        self.max_retries = max_retries
        self.retry_delay_base = retry_delay_base

        # Serper API configuration
        self.serper_server_addr = "https://google.serper.dev"
        self.serper_api_key = os.environ.get("SERPER_API_KEY", "")
        if not self.serper_api_key:
            raise RuntimeError(
                "Serper API key is not set. Please set the SERPER_API_KEY environment variable."
            )

        self.serper_headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        }

        logger.info(f"Initialized with Serper API key: {self.serper_api_key[:8]}...")


# ============== Async Online Search Client ==============

class AsyncOnlineSearchClient:
    """Async online search client using Serper API.

    This follows the same interface as Agent/Search-R1/search_r1/search/online_tools.py
    """

    _shared_session: Optional[aiohttp.ClientSession] = None
    _session_lock = threading.Lock()
    _search_semaphore: Optional[asyncio.Semaphore] = None

    @classmethod
    async def get_session(cls) -> aiohttp.ClientSession:
        """Get or create shared aiohttp session with connection pooling."""
        if cls._shared_session is None or cls._shared_session.closed:
            with cls._session_lock:
                if cls._shared_session is None or cls._shared_session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=1000,
                        limit_per_host=500,
                        ttl_dns_cache=600,
                        enable_cleanup_closed=True,
                    )
                    cls._shared_session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=aiohttp.ClientTimeout(total=300, sock_connect=100),
                        trust_env=True,
                    )
        return cls._shared_session

    @classmethod
    async def close_session(cls):
        """Close the shared session."""
        if cls._shared_session and not cls._shared_session.closed:
            await cls._shared_session.close()
            cls._shared_session = None

    @classmethod
    def _get_search_semaphore(cls) -> asyncio.Semaphore:
        """Get or create search semaphore for rate limiting."""
        if cls._search_semaphore is None:
            cls._search_semaphore = asyncio.Semaphore(20)
        return cls._search_semaphore

    def __init__(self, config: OnlineSearchConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    async def _do_serper_search(
        self,
        session: aiohttp.ClientSession,
        query: str,
        topk: int,
    ) -> Dict[str, Any]:
        """
        Execute a single Serper API search request.

        Args:
            session: aiohttp session
            query: Search query string (already truncated)
            topk: Number of results to return

        Returns:
            Dict with 'success' bool and either 'data' or 'error'

        Raises:
            Exception: If the request fails (to trigger retry)
        """
        async with self._get_search_semaphore():
            payload = {"q": query, "num": topk}
            await asyncio.sleep(0.1)  # Rate limiting

            async with session.post(
                f"{self.config.serper_server_addr}/search",
                headers=self.config.serper_headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.logger.info(data)
                    return {"success": True, "data": data}
                else:
                    response_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {response_text[:100]}")

    async def _single_serper_query(
        self,
        session: aiohttp.ClientSession,
        query: str,
        topk: int,
    ) -> Dict[str, Any]:
        """
        Execute a single Serper API search query with retry logic.

        Args:
            session: aiohttp session
            query: Search query string
            topk: Number of results to return

        Returns:
            Dict with 'success' bool and either 'data' or 'error'
        """
        query = query[:2000]  # Truncate long queries

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                if attempt > 0:
                    delay = self.config.retry_delay_base * (2 ** (attempt - 1)) + random.uniform(0, 20)
                    delay = min(delay, 300) + random.uniform(0, 20)
                    error_type = type(last_error).__name__ if last_error else "Unknown"
                    error_msg = str(last_error)[:100] if last_error else ""
                    if attempt > 5:
                        self.logger.warning(
                            f"Retrying search query '{query[:50]}...' "
                            f"(attempt {attempt + 1}/{self.config.max_retries}, delay {delay:.1f}s) "
                            f"- Last error: {error_type}: {error_msg}"
                        )
                    await asyncio.sleep(delay)

                return await self._do_serper_search(session, query, topk)

            except Exception as e:
                last_error = e
                if attempt == self.config.max_retries - 1:
                    error_msg = f"{type(e).__name__}: {str(e)[:200]}"
                    return {"success": False, "error": error_msg}

        return {"success": False, "error": "Unknown error after all retries"}

    async def query_async(self, req_meta: Dict[str, Any]) -> List[Dict]:
        """
        Query using Serper API with retry logic.

        This method follows the exact same interface as
        Agent/Search-R1/search_r1/search/online_tools.py::AsyncOnlineSearchClient.query_async

        Args:
            req_meta: Dict containing 'queries' list and 'topk' int

        Returns:
            List of dicts with 'documents', 'urls', and 'server_type'
        """
        queries = req_meta.get("queries", [])
        topk = req_meta.get("topk", self.config.topk)

        if not queries:
            return []

        session = await self.get_session()
        tasks = [self._single_serper_query(session, query, topk) for query in queries]
        serper_results = await asyncio.gather(*tasks)

        # Format results following the original online_tools.py logic
        formatted_results = []
        for query, serper_result in zip(queries, serper_results):
            if serper_result and serper_result.get("success", False):
                data = serper_result.get("data", {})
                organic_results = data.get("organic", [])[:topk]

                # Format: title + " " + snippet for documents
                documents = [
                    result.get("title", "") + " " + result.get("snippet", "")
                    for result in organic_results
                ]
                urls = [result.get("link", "") for result in organic_results]

                formatted_results.append({
                    "documents": documents,
                    "urls": urls,
                    "server_type": "async-online-search"
                })
            else:
                formatted_results.append({
                    "documents": [],
                    "urls": [],
                    "server_type": "async-online-search"
                })

        return formatted_results


# ============== FastAPI Application ==============

def create_app(config: OnlineSearchConfig) -> FastAPI:
    """Create FastAPI application with search endpoint."""

    app = FastAPI(
        title="Online Search Server",
        description="Online search server using Serper API, compatible with MASToolWorker interface",
        version="1.0.0",
    )

    search_client = AsyncOnlineSearchClient(config)

    @app.on_event("startup")
    async def startup_event():
        logger.info("Online search server starting...")
        logger.info(f"Configuration: topk={config.topk}, max_retries={config.max_retries}")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Online search server shutting down...")
        await AsyncOnlineSearchClient.close_session()

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "online-search"}

    @app.post("/retrieve")
    async def retrieve(request: SearchRequest):
        """
        Search endpoint compatible with MASToolWorker's query_async interface.

        Args:
            request: SearchRequest containing queries and optional topk

        Returns:
            Dict with 'result' containing list of search results
            Each result has 'documents', 'urls', and 'server_type'
        """
        try:
            req_meta = {
                "queries": request.queries,
                "topk": request.topk or config.topk,
                "return_scores": request.return_scores,
            }

            results = await search_client.query_async(req_meta)

            # Return in the same format as expected by infer_widesearch.py
            return {"result": results}

        except Exception as e:
            logger.error(f"Search error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ============== Main Entry Point ==============

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Online Search Server using Serper API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Server port",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Default number of search results per query",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=15,
        help="Maximum retries for failed requests",
    )
    parser.add_argument(
        "--retry_delay_base",
        type=float,
        default=5.0,
        help="Base delay (seconds) for exponential backoff",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn workers",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create configuration
    config = OnlineSearchConfig(
        topk=args.topk,
        max_retries=args.max_retries,
        retry_delay_base=args.retry_delay_base,
    )

    # Create and run application
    app = create_app(config)

    logger.info(f"Starting online search server on {args.host}:{args.port}")
    logger.info(f"API Documentation: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
