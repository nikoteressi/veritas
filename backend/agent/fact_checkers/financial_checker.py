import json
from typing import List, Dict, Any
from urllib.parse import urlparse

from .base import BaseFactChecker


class FinancialFactChecker(BaseFactChecker):
    """
    A fact-checker specializing in financial claims, prioritizing official sources.
    """
    role_description: str = (
        "A meticulous financial analyst who prioritizes official filings (e.g., SEC), "
        "market data from top-tier financial news, and regulatory statements."
    )

    def analyze_search_results(self, claim: str, search_results: List[Dict], temporal_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze search results from a financial perspective."""
        credible_domains = [
            "sec.gov", "federalreserve.gov", "treasury.gov", "bloomberg.com",
            "reuters.com", "wsj.com", "ft.com", "marketwatch.com", "cnbc.com",
            "forbes.com", "investopedia.com"
        ]
        
        verification_keywords = ["confirmed", "reported", "filed", "verified"]
        debunking_keywords = ["denied", "refuted", "corrected", "retracted"]
        
        total_results = 0
        credible_sources_count = 0
        supporting_evidence = 0
        contradicting_evidence = 0
        
        processed_sources = []

        for result_group in search_results:
            results_data = result_group.get("results", "")
            query = result_group.get("query", "")

            try:
                results_list = json.loads(results_data) if isinstance(results_data, str) else []
            except json.JSONDecodeError:
                continue

            for result in results_list:
                if not isinstance(result, dict):
                    continue
                
                total_results += 1
                url = result.get("url", "")
                title = result.get("title", "")
                content = result.get("content", "")
                
                domain = urlparse(url).netloc.lower().replace("www.", "")
                is_credible = any(cred_domain in domain for cred_domain in credible_domains)
                
                if is_credible:
                    credible_sources_count += 1
                
                content_lower = (title + " " + content).lower()
                verification_score = sum(1 for keyword in verification_keywords if keyword in content_lower)
                debunking_score = sum(1 for keyword in debunking_keywords if keyword in content_lower)

                if verification_score > debunking_score:
                    supporting_evidence += 1
                elif debunking_score > verification_score:
                    contradicting_evidence += 1
                
                processed_sources.append({
                    "url": url,
                    "title": title,
                    "domain": domain,
                    "credible": is_credible,
                    "verification_score": verification_score,
                    "debunking_score": debunking_score,
                    "query": query
                })

        # Confidence score calculation
        confidence_score = 0
        if credible_sources_count > 0:
            confidence_score += 40 * (credible_sources_count / max(total_results, 1))
        if supporting_evidence > contradicting_evidence:
            confidence_score += 25
        elif contradicting_evidence > supporting_evidence:
            confidence_score += 25
        
        preliminary_assessment = "unverified"
        if supporting_evidence > contradicting_evidence and credible_sources_count > 0:
            preliminary_assessment = "likely_true"
        elif contradicting_evidence > supporting_evidence:
            preliminary_assessment = "likely_false"
        elif total_results > 5 and credible_sources_count == 0:
            preliminary_assessment = "unverified_low_quality_sources"

        return {
            "claim": claim,
            "domain": "financial",
            "total_sources_found": total_results,
            "credible_sources": credible_sources_count,
            "fact_check_sources": 0, # Not the focus for financial checks
            "supporting_evidence": supporting_evidence,
            "contradicting_evidence": contradicting_evidence,
            "preliminary_assessment": preliminary_assessment,
            "confidence_score": min(confidence_score, 100),
            "processed_sources": processed_sources,
        } 