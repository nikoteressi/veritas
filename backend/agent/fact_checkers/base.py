from abc import ABC, abstractmethod
from typing import List, Dict, Any, TYPE_CHECKING
import logging
import json
import re

if TYPE_CHECKING:
    from ..tools import SearxNGSearchTool

from agent.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class BaseFactChecker(ABC):
    """Abstract base class for a domain-specific fact-checker."""

    role_description: str = "A generic fact-checker."
    prompt_manager = PromptManager()

    def __init__(self, search_tool: 'SearxNGSearchTool'):
        """
        Initializes the fact-checker with a search tool.
        
        Args:
            search_tool: An instance of SearxNGSearchTool to perform web searches.
        """
        self._search_tool = search_tool

    @abstractmethod
    async def analyze_search_results(self, claim: str, search_results: List[Dict[str, Any]], temporal_context: Dict[str, Any]) -> str:
        """
        Analyze search results from a domain-specific perspective.
        
        This method must be implemented by concrete checker classes. It should
        return a JSON string conforming to the FactCheckerResponse schema.
        """
        pass

    def _extract_sources_from_result(self, result: str) -> List[str]:
        """Utility to extract source URLs from search results."""
        if not isinstance(result, str):
            return []
            
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, result)

        clean_urls = []
        for url in urls:
            url = re.sub(r'[.,;:!?]+$', '', url)
            if url not in clean_urls:
                clean_urls.append(url)

        return clean_urls[:10]

    def _extract_entities(self, text: str) -> List[str]:
        """Utility to extract key entities (companies, people, organizations) from text."""
        entity_patterns = [
            r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b',
            r'\b(?:Inc|Corp|LLC|Ltd|Company|Group|Fund|Trust)\b',
            r'\b[A-Z]{2,}\b'
        ]

        entities = []
        for pattern in entity_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)

        common_words = {'The', 'This', 'That', 'And', 'But', 'For', 'With', 'On', 'At', 'By'}
        filtered_entities = [
            entity for entity in entities if len(entity) > 2 and entity not in common_words
        ]

        return list(dict.fromkeys(filtered_entities))[:5]

    async def check(self, claim: str, search_queries: List[str], temporal_context: Dict[str, Any]) -> str:
        """Execute the full fact-checking process for the given domain."""
        try:
            logger.info(f"Executing fact-check with {self.__class__.__name__}: '{self.role_description}'")
            
            search_results_data = []
            examined_sources = set()

            for query in search_queries:
                logger.info(f"Executing search query: {query}")
                result_str = await self._search_tool._arun(query=query)
                search_results_data.append({"query": query, "results": result_str})
                
                sources = self._extract_sources_from_result(result_str)
                examined_sources.update(sources)
                logger.info(f"Found {len(sources)} sources for query: '{query}'")

            # The analyze_search_results method is now async and returns a JSON string
            analysis_result_str = await self.analyze_search_results(claim, search_results_data, temporal_context)
            
            # We add the sources we found to the result
            analysis_result = json.loads(analysis_result_str)
            analysis_result["claim"] = claim
            analysis_result["examined_sources"] = list(examined_sources)
            analysis_result["total_sources_found"] = len(examined_sources)
            
            logger.info(f"Fact-checking completed for claim in {self.__class__.__name__}. Examined {len(examined_sources)} sources.")
            return json.dumps(analysis_result)

        except Exception as e:
            error_msg = f"Fact-checking failed in {self.__class__.__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            return json.dumps({"error": error_msg}) 