"""
from __future__ import annotations

Utility functions for graph verification.

Contains helper functions and common utilities used across verification modules.
"""

import asyncio
import hashlib
import logging
import re
import time
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class VerificationUtils:
    """Utility functions for verification operations."""

    @staticmethod
    def calculate_text_similarity(text1: str, text2: str) -> float:
        """Calculate simple text similarity using word overlap."""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def extract_keywords(text: str, min_length: int = 3) -> list[str]:
        """Extract keywords from text."""
        # Remove special characters and split
        words = re.findall(r"\b[a-zA-Z]{" + str(min_length) + ",}\b", text.lower())

        # Common stop words to filter out
        stop_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "this",
            "that",
            "these",
            "those",
            "was",
            "were",
            "been",
            "have",
            "has",
            "had",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "are",
            "is",
            "am",
        }

        return [word for word in words if word not in stop_words]

    @staticmethod
    def create_cache_key(*args) -> str:
        """Create a cache key from arguments."""
        key_string = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^\w\s.,!?;:-]", "", text)

        return text

    @staticmethod
    def truncate_text(text: str, max_length: int = 1000) -> str:
        """Truncate text to maximum length."""
        if len(text) <= max_length:
            return text

        # Try to truncate at sentence boundary
        truncated = text[:max_length]
        last_sentence = truncated.rfind(".")

        if last_sentence > max_length * 0.8:  # If we can keep 80% of text
            return truncated[: last_sentence + 1]

        return truncated + "..."

    @staticmethod
    def extract_domain(url: str) -> str | None:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return None

    @staticmethod
    def group_by_domain(urls: list[str]) -> dict[str, list[str]]:
        """Group URLs by domain."""
        domain_groups = {}

        for url in urls:
            domain = VerificationUtils.extract_domain(url)
            if domain:
                if domain not in domain_groups:
                    domain_groups[domain] = []
                domain_groups[domain].append(url)

        return domain_groups

    @staticmethod
    def calculate_confidence_score(evidence_count: int, source_diversity: int, consistency_score: float) -> float:
        """Calculate confidence score based on multiple factors."""
        # Base score from evidence count (diminishing returns)
        evidence_score = min(1.0, evidence_count / 5.0)

        # Source diversity bonus
        diversity_score = min(1.0, source_diversity / 3.0)

        # Weighted combination
        confidence = evidence_score * 0.4 + diversity_score * 0.3 + consistency_score * 0.3

        return min(1.0, max(0.0, confidence))

    @staticmethod
    def detect_contradictory_statements(statements: list[str]) -> list[tuple[int, int]]:
        """Detect potentially contradictory statements using simple heuristics."""
        contradictions = []

        # Simple contradiction patterns
        negation_patterns = [
            (r"\bnot\b", r"\bis\b"),
            (r"\bno\b", r"\byes\b"),
            (r"\bfalse\b", r"\btrue\b"),
            (r"\bdeny\b", r"\bconfirm\b"),
            (r"\brefute\b", r"\bsupport\b"),
        ]

        for i, stmt1 in enumerate(statements):
            for j, stmt2 in enumerate(statements[i + 1 :], i + 1):
                # Check for direct negation patterns
                for neg_pattern, pos_pattern in negation_patterns:
                    if (re.search(neg_pattern, stmt1.lower()) and re.search(pos_pattern, stmt2.lower())) or (
                        re.search(pos_pattern, stmt1.lower()) and re.search(neg_pattern, stmt2.lower())
                    ):
                        contradictions.append((i, j))
                        break

        return contradictions

    @staticmethod
    async def batch_process_with_semaphore(
        items: list[Any],
        process_func,
        max_concurrent: int = 5,
        delay_between_batches: float = 0.1,
    ) -> list[Any]:
        """Process items in batches with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []

        async def process_item(item):
            async with semaphore:
                try:
                    result = await process_func(item)
                    await asyncio.sleep(delay_between_batches)
                    return result
                except Exception as e:
                    logger.error(f"Error processing item {item}: {e}")
                    return None

        tasks = [process_item(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results and exceptions
        return [r for r in results if r is not None and not isinstance(r, Exception)]

    @staticmethod
    def measure_execution_time(func):
        """Decorator to measure function execution time."""

        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result

        return wrapper

    @staticmethod
    async def measure_async_execution_time(func):
        """Decorator to measure async function execution time."""

        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result

        return wrapper

    @staticmethod
    def safe_get_nested(data: dict, keys: list[str], default=None):
        """Safely get nested dictionary value."""
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    @staticmethod
    def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(VerificationUtils.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def deduplicate_by_similarity(texts: list[str], similarity_threshold: float = 0.8) -> list[str]:
        """Remove duplicate texts based on similarity."""
        if not texts:
            return []

        unique_texts = [texts[0]]

        for text in texts[1:]:
            is_duplicate = False
            for unique_text in unique_texts:
                similarity = VerificationUtils.calculate_text_similarity(text, unique_text)
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_texts.append(text)

        return unique_texts

    @staticmethod
    def create_search_variations(query: str) -> list[str]:
        """Create variations of search query for better coverage."""
        variations = [query]

        # Add quoted version for exact phrase
        if " " in query and not query.startswith('"'):
            variations.append(f'"{query}"')

        # Add question variations
        question_words = ["what", "when", "where", "who", "why", "how"]
        for qword in question_words:
            if qword not in query.lower():
                variations.append(f"{qword} {query}")

        # Add verification-specific terms
        verification_terms = ["verify", "fact check", "evidence", "proof"]
        for term in verification_terms:
            if term not in query.lower():
                variations.append(f"{query} {term}")

        return variations[:5]  # Limit to 5 variations

    @staticmethod
    def extract_numbers_and_dates(text: str) -> dict[str, list[str]]:
        """Extract numbers and dates from text for fact checking."""
        # Extract numbers
        numbers = re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", text)

        # Extract dates (simple patterns)
        date_patterns = [
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",  # MM/DD/YYYY
            r"\b\d{4}-\d{1,2}-\d{1,2}\b",  # YYYY-MM-DD
            r"\b\w+ \d{1,2}, \d{4}\b",  # Month DD, YYYY
            r"\b\d{1,2} \w+ \d{4}\b",  # DD Month YYYY
        ]

        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text))

        return {"numbers": numbers, "dates": dates}


# CacheManager has been removed - use unified cache system from app.cache.factory instead
