#from together_open_deep_research
import asyncio
import os
import sys
import select
from dataclasses import dataclass
from typing import Optional

from tavily import AsyncTavilyClient, TavilyClient

from dataclasses import dataclass
from typing import List
from pydantic import BaseModel, Field

@dataclass(frozen=True, kw_only=True)
class SearchResult:
    title: str
    link: str
    content: str
    raw_content: Optional[str] = None

    def __str__(self, include_raw=True):
        result = f"Title: {self.title}\n" f"Link: {self.link}\n" f"Content: {self.content}"
        if include_raw and self.raw_content:
            result += f"\nRaw Content: {self.raw_content}"
        return result

    def short_str(self):
        return self.__str__(include_raw=False)


@dataclass(frozen=True, kw_only=True)
class SearchResults:
    results: list[SearchResult]

    def __str__(self, short=False):
        if short:
            result_strs = [result.short_str() for result in self.results]
        else:
            result_strs = [str(result) for result in self.results]
        return "\n\n".join(f"[{i+1}] {result_str}" for i, result_str in enumerate(result_strs))

    def __add__(self, other):
        return SearchResults(results=self.results + other.results)

    def short_str(self):
        return self.__str__(short=True)


def extract_tavily_results(response) -> SearchResults:
    """Extract key information from Tavily search results."""
    results = []
    for item in response.get("results", []):
        results.append(
            SearchResult(
                title=item.get("title", ""),
                link=item.get("url", ""),
                content=item.get("content", ""),
                raw_content=item.get("raw_content", ""),
            )
        )
    return SearchResults(results=results)


def tavily_search(query: str, max_results=3, include_raw: bool = True) -> SearchResults:
    """
    Perform a search using the Tavily Search API with the official client.

    Parameters:
        query (str): The search query.
        search_depth (str): The depth of search - 'basic' or 'deep'.
        max_results (int): Maximum number of results to return.

    Returns:
        list: Formatted search results with title, link, and snippet.
    """
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")

    client = TavilyClient(api_key)

    response = client.search(query=query, search_depth="basic", max_results=max_results, include_raw_content=include_raw)

    return extract_tavily_results(response)


async def atavily_search_results(query: str, max_results=3, include_raw: bool = True) -> SearchResults:
    """
    Perform asynchronous search using the Tavily Search API with the official client.

    Parameters:
        query (str): The search query.
        max_results (int): Maximum number of results to return.
    """
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")

    client = AsyncTavilyClient(api_key)

    response = await client.search(query=query, search_depth="basic", max_results=max_results, include_raw_content=include_raw)

    return extract_tavily_results(response)

class ResearchPlan(BaseModel):
    queries: list[str] = Field(
        description="A list of search queries to thoroughly research the topic")


class SourceList(BaseModel):
    sources: list[int] = Field(
        description="A list of source numbers from the search results")

class UserCommunication:
    """Handles user input/output interactions with timeout functionality."""

    @staticmethod
    async def get_input_with_timeout(prompt: str, timeout: float = 30.0) -> str:
        """
        Get user input with a timeout.
        Returns empty string if timeout occurs or no input is provided.

        Args:
            prompt: The prompt to display to the user
            timeout: Number of seconds to wait for user input (default: 30.0)

        Returns:
            str: User input or empty string if timeout occurs
        """
        print(prompt, end="", flush=True)

        # Different implementation for Windows vs Unix-like systems
        if sys.platform == "win32":
            # Windows implementation
            try:
                # Run input in an executor to make it async
                loop = asyncio.get_event_loop()
                user_input = await asyncio.wait_for(loop.run_in_executor(None, input), timeout)
                return user_input.strip()
            except TimeoutError:
                print("\nTimeout reached, continuing...")
                return ""
        else:
            # Unix-like implementation
            i, _, _ = select.select([sys.stdin], [], [], timeout)
            if i:
                return sys.stdin.readline().strip()
            else:
                print("\nTimeout reached, continuing...")
                return ""

@dataclass(frozen=True, kw_only=True)
class DeepResearchResult(SearchResult):
    """Wrapper on top of SearchResults to adapt it to the DeepResearch.

    This class extends the basic SearchResult by adding a filtered version of the raw content
    that has been processed and refined for the specific research context. It maintains
    the original search result while providing additional research-specific information.

    Attributes:
        filtered_raw_content: A processed version of the raw content that has been filtered
                             and refined for relevance to the research topic
    """

    filtered_raw_content: str

    def __str__(self):
        return f"Title: {self.title}\n" f"Link: {self.link}\n" f"Refined Content: {self.filtered_raw_content[:10000]}"

    def short_str(self):
        return f"Title: {self.title}\nLink: {self.link}\nRaw Content: {self.content[:10000]}"


@dataclass(frozen=True, kw_only=True)
class DeepResearchResults(SearchResults):
    results: list[DeepResearchResult]

    def __add__(self, other):
        return DeepResearchResults(results=self.results + other.results)

    def dedup(self):
        def deduplicate_by_link(results):
            seen_links = set()
            unique_results = []

            for result in results:
                if result.link not in seen_links:
                    seen_links.add(result.link)
                    unique_results.append(result)

            return unique_results

        return DeepResearchResults(results=deduplicate_by_link(self.results)) 