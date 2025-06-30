import json
from typing import List, Dict, Any
from urllib.parse import urlparse

from .base import BaseFactChecker


class GeneralFactChecker(BaseFactChecker):
    """
    A general-purpose fact-checker for a wide variety of claims.
    """
    role_description: str = (
        "A versatile fact-checker that verifies general claims by cross-referencing "
        "reputable news sources and established fact-checking organizations."
    )

    def analyze_search_results(self, claim: str, search_results: List[Dict], temporal_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze search results for general claims."""
        credible_domains = [
            "reuters.com", "ap.org", "bbc.com", "npr.org", "factcheck.org", 
            "snopes.com", "politifact.com", "nytimes.com", "washingtonpost.com"
        ]
        
        verification_keywords = ["fact check", "verified", "confirmed", "true", "accurate"]
        debunking_keywords = ["false", "debunked", "misleading", "incorrect", "fake", "hoax"]
        
        total_results = 0
        credible_sources_count = 0
        fact_check_sources_count = 0
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
            confidence_score += 30 * (credible_sources_count / max(total_results, 1))
        if supporting_evidence > contradicting_evidence:
            confidence_score += 20
        elif contradicting_evidence > supporting_evidence:
            confidence_score += 20
        
        preliminary_assessment = "unverified"
        if supporting_evidence > contradicting_evidence:
            preliminary_assessment = "likely_true"
        elif contradicting_evidence > supporting_evidence:
            preliminary_assessment = "likely_false"

        return {
            "claim": claim,
            "domain": "general",
            "total_sources_found": total_results,
            "credible_sources": credible_sources_count,
            "fact_check_sources": fact_check_sources_count,
            "supporting_evidence": supporting_evidence,
            "contradicting_evidence": contradicting_evidence,
            "preliminary_assessment": preliminary_assessment,
            "confidence_score": min(confidence_score, 100),
            "processed_sources": processed_sources,
        } 