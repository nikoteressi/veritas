"""
Custom tools for the LangChain agent.
"""
import json
import logging
from typing import Dict, Any, Optional, List

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import AsyncSessionLocal
from app.crud import UserCRUD

logger = logging.getLogger(__name__)


class SearxNGSearchInput(BaseModel):
    """Input schema for SearxNG search tool."""
    query: str = Field(description="Search query to execute")
    category: Optional[str] = Field(default="general", description="Search category (general, news, science, etc.)")
    engines: Optional[str] = Field(default=None, description="Specific search engines to use")
    language: Optional[str] = Field(default="en", description="Search language")


class SearxNGSearchTool(BaseTool):
    """Tool for searching the web using SearxNG."""

    name: str = "searxng_search"
    description: str = """
    Search the web for information using SearxNG search engine.
    Use this tool to find current information, verify facts, and gather evidence.
    Provide a clear, specific search query for best results.
    """
    args_schema: type = SearxNGSearchInput
    
    def _run(self, query: str, category: str = "general", engines: str = None, language: str = "en") -> str:
        """Execute the search synchronously."""
        try:
            # Prepare search parameters
            params = {
                "q": query,
                "format": "json",
                "category": category,
                "language": language
            }
            
            if engines:
                params["engines"] = engines
            
            # Make request to SearxNG
            response = requests.get(
                f"{settings.searxng_url}/search",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                return f"No search results found for query: {query}"
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results[:5]):  # Limit to top 5 results
                formatted_result = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "engine": result.get("engine", "")
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"SearxNG search completed: {len(formatted_results)} results for '{query}'")
            return json.dumps(formatted_results, indent=2)
            
        except Exception as e:
            error_msg = f"SearxNG search failed: {e}"
            logger.error(error_msg)
            return error_msg
    
    async def _arun(self, query: str, category: str = "general", engines: str = None, language: str = "en") -> str:
        """Execute the search asynchronously."""
        # For now, use the sync version
        # In production, you might want to use aiohttp for async requests
        return self._run(query, category, engines, language)


class DatabaseReputationInput(BaseModel):
    """Input schema for database reputation tool."""
    nickname: str = Field(description="User nickname to look up or update")
    action: str = Field(description="Action to perform: 'get' or 'update'")
    verdict: Optional[str] = Field(default=None, description="Verdict for update: true, partially_true, false, ironic")


class DatabaseReputationTool(BaseTool):
    """Tool for interacting with user reputation database."""

    name: str = "database_reputation"
    description: str = """
    Get or update user reputation information in the database.
    Use 'get' action to retrieve current reputation data.
    Use 'update' action to update reputation after verification (requires verdict).
    """
    args_schema: type = DatabaseReputationInput
    
    def _run(self, nickname: str, action: str, verdict: str = None) -> str:
        """Execute database operation synchronously."""
        # This is a simplified sync version
        # In practice, you'd want to handle this differently
        return f"Database operation: {action} for {nickname} (verdict: {verdict})"
    
    async def _arun(self, nickname: str, action: str, verdict: str = None) -> str:
        """Execute database operation asynchronously."""
        try:
            async with AsyncSessionLocal() as db:
                if action == "get":
                    user = await UserCRUD.get_user_by_nickname(db, nickname)
                    if user:
                        reputation_data = {
                            "nickname": user.nickname,
                            "true_count": user.true_count,
                            "partially_true_count": user.partially_true_count,
                            "false_count": user.false_count,
                            "ironic_count": user.ironic_count,
                            "total_posts_checked": user.total_posts_checked,
                            "warning_issued": user.warning_issued,
                            "notification_issued": user.notification_issued
                        }
                        logger.info(f"Retrieved reputation for {nickname}")
                        return json.dumps(reputation_data, indent=2)
                    else:
                        return f"No reputation data found for user: {nickname}"
                
                elif action == "update" and verdict:
                    user = await UserCRUD.update_user_reputation(db, nickname, verdict)
                    result = {
                        "nickname": user.nickname,
                        "verdict": verdict,
                        "new_total": user.total_posts_checked,
                        "warning_issued": user.warning_issued,
                        "notification_issued": user.notification_issued
                    }
                    logger.info(f"Updated reputation for {nickname}: {verdict}")
                    return json.dumps(result, indent=2)
                
                else:
                    return f"Invalid action or missing verdict: {action}"
                    
        except Exception as e:
            error_msg = f"Database operation failed: {e}"
            logger.error(error_msg)
            return error_msg


class FactCheckingInput(BaseModel):
    """Input schema for fact-checking tool."""
    claim: str = Field(description="Specific claim to fact-check")
    context: Optional[str] = Field(default=None, description="Additional context for the claim")
    domain: Optional[str] = Field(default="general", description="Domain of the claim (medical, financial, etc.)")


class FactCheckingTool(BaseTool):
    """Tool for specialized fact-checking based on domain."""

    name: str = "fact_checking"
    description: str = """
    Perform specialized fact-checking for specific claims.
    This tool combines search results with domain-specific knowledge to verify claims.
    Specify the domain (medical, financial, scientific, political, general) for better accuracy.
    """
    args_schema: type = FactCheckingInput
    
    def __init__(self, search_tool: SearxNGSearchTool = None):
        super().__init__()
        self._search_tool = search_tool or SearxNGSearchTool()
    
    def _run(self, claim: str, context: str = None, domain: str = "general") -> str:
        """Perform fact-checking synchronously with enhanced temporal and source analysis."""
        try:
            # Parse context for temporal information
            temporal_context = self._parse_temporal_context(context)

            # Create domain-specific search queries with temporal awareness
            search_queries = self._generate_search_queries(claim, domain, temporal_context)

            # Perform searches with detailed logging
            search_results = []
            examined_sources = []

            for query in search_queries:
                logger.info(f"Executing search query: {query}")
                result = self._search_tool._run(query)
                search_results.append({"query": query, "results": result})

                # Extract source URLs for tracking
                sources = self._extract_sources_from_result(result)
                examined_sources.extend(sources)
                logger.info(f"Found {len(sources)} sources for query: {query}")

            # Analyze results with temporal and source context
            analysis = self._analyze_search_results(claim, search_results, domain, temporal_context)
            analysis["examined_sources"] = examined_sources
            analysis["search_queries_used"] = search_queries

            logger.info(f"Fact-checking completed for claim in {domain} domain. "
                       f"Examined {len(examined_sources)} sources.")
            return json.dumps(analysis, indent=2)

        except Exception as e:
            error_msg = f"Fact-checking failed: {e}"
            logger.error(error_msg)
            return error_msg
    
    async def _arun(self, claim: str, context: str = None, domain: str = "general") -> str:
        """Perform fact-checking asynchronously."""
        return self._run(claim, context, domain)
    
    def _parse_temporal_context(self, context: str) -> Dict[str, Any]:
        """Parse temporal context from the provided context string."""
        if not context:
            return {}

        try:
            # Try to parse as JSON first (if context contains temporal analysis)
            import json
            context_data = json.loads(context)
            return context_data.get("temporal_analysis", {})
        except:
            # Fallback to simple parsing
            return {"raw_context": context}

    def _extract_sources_from_result(self, result: str) -> List[str]:
        """Extract source URLs from search results."""
        import re

        # Extract URLs from the result string
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, result)

        # Clean and deduplicate URLs
        clean_urls = []
        for url in urls:
            # Remove trailing punctuation
            url = re.sub(r'[.,;:!?]+$', '', url)
            if url not in clean_urls:
                clean_urls.append(url)

        return clean_urls[:10]  # Limit to 10 sources per query

    def _generate_search_queries(self, claim: str, domain: str, temporal_context: Dict[str, Any] = None) -> List[str]:
        """Generate sophisticated, multi-perspective search queries with temporal awareness."""
        import re

        # Extract key entities and numbers from the claim
        entities = self._extract_entities(claim)
        numbers = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', claim)

        # Base verification queries
        base_queries = [
            f'"{claim}" fact check verification',
            f'"{claim}" debunked false true',
            f"{claim} evidence sources"
        ]

        # Multi-perspective queries
        perspective_queries = [
            f"{claim} contradictory evidence opposing view",
            f"{claim} alternative explanation different perspective",
            f"{claim} context background full story"
        ]

        # Temporal-aware queries
        temporal_queries = []
        if temporal_context and temporal_context.get("temporal_mismatch"):
            temporal_queries.extend([
                f"{claim} outdated information old news",
                f"{claim} when did this happen date verification",
                f"{claim} timeline chronology"
            ])

        # Entity-specific queries
        entity_queries = []
        for entity in entities[:2]:  # Limit to 2 main entities
            entity_queries.extend([
                f'"{entity}" recent news updates',
                f'"{entity}" official statement response'
            ])

        # Number/statistic verification
        number_queries = []
        if numbers:
            for number in numbers[:2]:  # Limit to 2 main numbers
                number_queries.extend([
                    f'"{number}" statistic verification accuracy',
                    f"{number} data source methodology"
                ])

        # Domain-specific sophisticated queries
        domain_specific = {
            "financial": [
                f"{claim} SEC filing official data",
                f"{claim} financial report quarterly earnings",
                f"{claim} market analysis expert opinion",
                f"{claim} regulatory filing disclosure"
            ],
            "medical": [
                f"{claim} peer reviewed study clinical trial",
                f"{claim} medical journal publication",
                f"{claim} FDA approval clinical evidence",
                f"{claim} medical expert consensus"
            ],
            "scientific": [
                f"{claim} peer reviewed research paper",
                f"{claim} scientific journal publication",
                f"{claim} research methodology data",
                f"{claim} scientific consensus expert review"
            ],
            "political": [
                f"{claim} government official statement",
                f"{claim} public record official document",
                f"{claim} policy analysis expert opinion",
                f"{claim} legislative record voting history"
            ]
        }

        # Combine all query types
        all_queries = base_queries + perspective_queries

        if domain in domain_specific:
            all_queries.extend(domain_specific[domain][:3])

        all_queries.extend(entity_queries[:2])
        all_queries.extend(number_queries[:2])

        # Remove duplicates and limit total queries
        unique_queries = list(dict.fromkeys(all_queries))
        return unique_queries[:6]  # Increased from 3 to 6 for better coverage

    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities (companies, people, organizations) from text."""
        import re

        # Simple entity extraction patterns
        entity_patterns = [
            r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b',  # Capitalized words/phrases
            r'\b(?:Inc|Corp|LLC|Ltd|Company|Group|Fund|Trust)\b',  # Company suffixes
            r'\b[A-Z]{2,}\b'  # Acronyms
        ]

        entities = []
        for pattern in entity_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)

        # Filter out common words and short entities
        filtered_entities = []
        common_words = {'The', 'This', 'That', 'And', 'But', 'For', 'With', 'On', 'At', 'By'}

        for entity in entities:
            if len(entity) > 2 and entity not in common_words:
                filtered_entities.append(entity)

        return list(dict.fromkeys(filtered_entities))[:5]  # Remove duplicates, limit to 5
    
    def _analyze_search_results(self, claim: str, search_results: List[Dict], domain: str, temporal_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze search results with enhanced credibility, temporal, and content assessment."""
        import json
        from urllib.parse import urlparse

        # Credible source domains by category
        credible_domains = {
            "financial": [
                "sec.gov", "federalreserve.gov", "treasury.gov", "bloomberg.com",
                "reuters.com", "wsj.com", "ft.com", "marketwatch.com", "cnbc.com"
            ],
            "medical": [
                "nih.gov", "cdc.gov", "who.int", "fda.gov", "nejm.org",
                "thelancet.com", "bmj.com", "pubmed.ncbi.nlm.nih.gov"
            ],
            "scientific": [
                "nature.com", "science.org", "pnas.org", "arxiv.org",
                "pubmed.ncbi.nlm.nih.gov", "ieee.org"
            ],
            "political": [
                "congress.gov", "whitehouse.gov", "supremecourt.gov",
                "politifact.com", "factcheck.org", "snopes.com"
            ],
            "general": [
                "reuters.com", "ap.org", "bbc.com", "npr.org",
                "factcheck.org", "snopes.com", "politifact.com"
            ]
        }

        total_results = 0
        credible_sources = 0
        temporal_flags = []
        fact_check_sources = 0
        supporting_evidence = 0
        contradicting_evidence = 0

        # Keywords that suggest verification/debunking
        verification_keywords = ["fact check", "verified", "confirmed", "true", "accurate"]
        debunking_keywords = ["false", "debunked", "misleading", "incorrect", "fake"]

        processed_results = []

        for result_group in search_results:
            query = result_group.get("query", "")
            results_data = result_group.get("results", "")

            # Parse results if they're in string format
            if isinstance(results_data, str):
                try:
                    results_list = json.loads(results_data)
                except:
                    continue
            else:
                results_list = results_data if isinstance(results_data, list) else []

            for result in results_list:
                if not isinstance(result, dict):
                    continue

                total_results += 1
                url = result.get("url", "")
                title = result.get("title", "")
                content = result.get("content", "")

                # Check source credibility
                domain = urlparse(url).netloc.lower()
                domain_credible = False

                for category, domains in credible_domains.items():
                    if any(cred_domain in domain for cred_domain in domains):
                        credible_sources += 1
                        domain_credible = True
                        break

                # Check if it's a fact-checking source
                fact_check_indicators = ["factcheck", "snopes", "politifact", "verification"]
                if any(indicator in domain for indicator in fact_check_indicators):
                    fact_check_sources += 1

                # Analyze content sentiment toward the claim
                content_lower = (title + " " + content).lower()

                verification_score = sum(1 for keyword in verification_keywords if keyword in content_lower)
                debunking_score = sum(1 for keyword in debunking_keywords if keyword in content_lower)

                if verification_score > debunking_score:
                    supporting_evidence += 1
                elif debunking_score > verification_score:
                    contradicting_evidence += 1

                processed_results.append({
                    "url": url,
                    "title": title,
                    "domain": domain,
                    "credible": domain_credible,
                    "verification_score": verification_score,
                    "debunking_score": debunking_score,
                    "query": query
                })

        # Calculate confidence based on multiple factors
        confidence_score = 0
        confidence_factors = []

        if credible_sources > 0:
            credibility_ratio = credible_sources / max(total_results, 1)
            confidence_score += credibility_ratio * 30
            confidence_factors.append(f"Credible sources: {credible_sources}/{total_results}")

        if fact_check_sources > 0:
            confidence_score += min(fact_check_sources * 15, 25)
            confidence_factors.append(f"Fact-check sources: {fact_check_sources}")

        if supporting_evidence > 0 or contradicting_evidence > 0:
            evidence_clarity = abs(supporting_evidence - contradicting_evidence) / max(supporting_evidence + contradicting_evidence, 1)
            confidence_score += evidence_clarity * 25
            confidence_factors.append(f"Evidence clarity: {evidence_clarity:.2f}")

        if total_results >= 10:
            confidence_score += 10
            confidence_factors.append("Sufficient sources found")

        # Add temporal analysis to confidence assessment
        if temporal_context:
            if temporal_context.get("temporal_mismatch"):
                severity = temporal_context.get("mismatch_severity", "none")
                if severity == "critical":
                    confidence_score -= 30
                    temporal_flags.append("Critical temporal mismatch detected")
                elif severity == "major":
                    confidence_score -= 20
                    temporal_flags.append("Major temporal mismatch detected")
                elif severity == "minor":
                    confidence_score -= 10
                    temporal_flags.append("Minor temporal mismatch detected")

                # Check for potential manipulation intent
                intent = temporal_context.get("intent_analysis", "unknown")
                if intent == "potential_market_manipulation":
                    confidence_score -= 25
                    temporal_flags.append("Potential market manipulation detected")

        # Determine preliminary assessment
        if contradicting_evidence > supporting_evidence and fact_check_sources > 0:
            preliminary_assessment = "likely_false"
        elif supporting_evidence > contradicting_evidence and credible_sources > 2:
            preliminary_assessment = "likely_true"
        elif total_results < 3:
            preliminary_assessment = "insufficient_evidence"
        elif temporal_flags:
            preliminary_assessment = "temporal_mismatch_detected"
        else:
            preliminary_assessment = "requires_human_review"

        analysis = {
            "claim": claim,
            "domain": domain,
            "search_queries_used": [r["query"] for r in search_results],
            "total_sources_found": total_results,
            "credible_sources": credible_sources,
            "fact_check_sources": fact_check_sources,
            "supporting_evidence": supporting_evidence,
            "contradicting_evidence": contradicting_evidence,
            "preliminary_assessment": preliminary_assessment,
            "confidence_score": max(min(int(confidence_score), 95), 0),  # Cap between 0-95%
            "confidence_factors": confidence_factors,
            "temporal_analysis": temporal_context or {},
            "temporal_flags": temporal_flags,
            "processed_sources": processed_results[:10],  # Limit to top 10 for brevity
            "sources": search_results
        }

        return analysis


# Initialize tools
searxng_tool = SearxNGSearchTool()
database_tool = DatabaseReputationTool()
fact_checking_tool = FactCheckingTool()

# Export tools list
AVAILABLE_TOOLS = [
    searxng_tool,
    database_tool,
    fact_checking_tool
]
