from abc import ABC, abstractmethod
from typing import List, Dict, Any, TYPE_CHECKING, Optional
import logging
import json
import re
import asyncio

if TYPE_CHECKING:
    from ..tools import SearxNGSearchTool

from agent.prompt_manager import PromptManager
from agent.analyzers.temporal_analyzer import TemporalAnalysisResult
from agent.models.internal import FactCheckerResponse, CredibleSource, SourceSelectionResponse
from langchain.output_parsers import PydanticOutputParser

logger = logging.getLogger(__name__)


class BaseFactChecker(ABC):
    """Abstract base class for a domain-specific fact-checker."""

    role_description: str = "A generic fact-checker."
    domain: str = "general"  # Default domain
    prompt_manager = PromptManager()

    def __init__(self, search_tool: 'SearxNGSearchTool', domain: str = None):
        """
        Initializes the fact-checker with a search tool.
        
        Args:
            search_tool: An instance of SearxNGSearchTool to perform web searches.
            domain: The domain for this fact-checker (e.g., 'financial', 'general')
        """
        self._search_tool = search_tool
        if domain:
            self.domain = domain
        # Get domain-specific role description from prompts.yaml
        self.role_description = self.prompt_manager.get_domain_role_description(self.domain)

    @abstractmethod
    async def analyze_search_results(self, claim: str, search_results: List[Dict[str, Any]], temporal_context: Optional[TemporalAnalysisResult]) -> FactCheckerResponse:
        """
        Analyze search results from a domain-specific perspective.
        
        This method must be implemented by concrete checker classes. It should
        return a FactCheckerResponse object.
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

    async def _select_credible_sources(self, claim: str, search_results: List[Dict[str, Any]]) -> List[str]:
        """
        Uses an LLM to select the most credible and relevant sources from search results.
        
        Args:
            claim: The claim being fact-checked
            search_results: List of search result dictionaries
            
        Returns:
            List of selected credible URLs
            
        Raises:
            ValueError: If the LLM response is invalid or contains no URLs.
        """
        from agent.llm import llm_manager
        
        logger.info("Selecting credible sources using LLM with domain expertise.")
        parser = PydanticOutputParser(pydantic_object=SourceSelectionResponse)
        
        try:
            prompt_template = self.prompt_manager.get_prompt_template("source_selection")
            prompt = await prompt_template.aformat(
                role_description=self.role_description,
                claim=claim,
                search_results=json.dumps(search_results),
                format_instructions=parser.get_format_instructions(),
            )
            response = await llm_manager.invoke_text_only(prompt)
            parsed_response = parser.parse(response)

            if not parsed_response.credible_urls:
                raise ValueError("LLM response for source selection contains no URLs.")

            logger.info(f"Selected {len(parsed_response.credible_urls)} credible sources. Reasoning: {parsed_response.reasoning}")
            return parsed_response.credible_urls
        except Exception as e:
            logger.error(f"Failed to select credible sources for claim '{claim}': {e}")
            raise

    async def check(self, claim: str, search_queries: List[str], temporal_context: Optional[TemporalAnalysisResult]) -> FactCheckerResponse:
        """Execute the full fact-checking process for the given domain."""
        try:
            logger.info(f"Executing fact-check with {self.__class__.__name__}: '{self.role_description}'")
            
            # Execute all search queries in parallel
            logger.info(f"Executing {len(search_queries)} search queries in parallel")
            search_tasks = [
                self._search_tool._arun(query=query) 
                for query in search_queries
            ]
            
            # Wait for all searches to complete
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Process results and collect sources
            search_results_data = []
            examined_sources = set()
            
            for i, (query, result) in enumerate(zip(search_queries, search_results)):
                if isinstance(result, Exception):
                    logger.error(f"Search query '{query}' failed: {result}")
                    # Add empty result for failed queries
                    search_results_data.append({"query": query, "results": f"Search failed: {result}"})
                else:
                    search_results_data.append({"query": query, "results": result})
                    sources = self._extract_sources_from_result(result)
                    examined_sources.update(sources)
                    logger.info(f"Found {len(sources)} sources for query: '{query}'")
                    logger.info(f"GRABBED SOURSES: {sources}")

            # The analyze_search_results method now returns a FactCheckerResponse object
            analysis_result = await self.analyze_search_results(claim, search_results_data, temporal_context)
            
            logger.info(f"Fact-checking completed for claim in {self.__class__.__name__}. Examined {len(examined_sources)} sources.")
            return analysis_result

        except Exception as e:
            error_msg = f"Fact-checking failed in {self.__class__.__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            return FactCheckerResponse(
                assessment="error",
                summary=error_msg,
                confidence=0.0,
                supporting_evidence=0,
                contradicting_evidence=0,
                credible_sources=[]
            )