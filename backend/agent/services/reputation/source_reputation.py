"""
Advanced source reputation and credibility assessment system.

This module provides comprehensive evaluation of information sources
including historical accuracy tracking, bias analysis, and cross-referencing
with known reliable/unreliable source databases.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import Settings

settings = Settings()


class SourceType(Enum):
    """Types of information sources."""

    NEWS_MEDIA = "news_media"
    ACADEMIC = "academic"
    GOVERNMENT = "government"
    SOCIAL_MEDIA = "social_media"
    BLOG = "blog"
    WIKI = "wiki"
    FORUM = "forum"
    UNKNOWN = "unknown"


class BiasDirection(Enum):
    """Political/ideological bias directions."""

    LEFT = "left"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    RIGHT = "right"
    UNKNOWN = "unknown"


@dataclass
class SourceMetrics:
    """Metrics for source evaluation."""

    accuracy_score: float = 0.0  # Historical accuracy (0-1)
    reliability_score: float = 0.0  # Overall reliability (0-1)
    bias_score: float = 0.5  # Bias level (0=very biased, 1=neutral)
    bias_direction: BiasDirection = BiasDirection.UNKNOWN
    transparency_score: float = 0.0  # Source transparency (0-1)
    expertise_score: float = 0.0  # Domain expertise (0-1)
    recency_score: float = 0.0  # How recent/current (0-1)
    citation_score: float = 0.0  # Quality of citations (0-1)

    verification_count: int = 0
    correct_predictions: int = 0
    false_predictions: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class SourceProfile:
    """Complete profile of an information source."""

    domain: str
    source_type: SourceType
    metrics: SourceMetrics
    metadata: dict[str, Any] = field(default_factory=dict)

    # Historical data
    verification_history: list[dict[str, Any]] = field(default_factory=list)
    bias_indicators: list[str] = field(default_factory=list)

    # External ratings
    external_ratings: dict[str, float] = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class SourceReputationSystem:
    """
    Advanced source reputation and credibility assessment system.

    Evaluates sources based on multiple factors including historical accuracy,
    bias analysis, transparency, expertise, and cross-referencing with
    known reliable/unreliable source databases.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.source_profiles: dict[str, SourceProfile] = {}

        # Known reliable sources (can be loaded from external database)
        self.reliable_sources = {
            "reuters.com": 0.95,
            "apnews.com": 0.94,
            "bbc.com": 0.92,
            "npr.org": 0.91,
            "nature.com": 0.98,
            "science.org": 0.97,
            "nejm.org": 0.96,
            "who.int": 0.93,
            "cdc.gov": 0.92,
            "gov.uk": 0.88,
            "europa.eu": 0.87,
        }

        # Known unreliable sources
        self.unreliable_sources = {
            "infowars.com": 0.1,
            "naturalnews.com": 0.15,
            "beforeitsnews.com": 0.2,
            "worldnewsdailyreport.com": 0.1,
            "theonion.com": 0.0,  # Satire
        }

        # Bias indicators
        self.bias_keywords = {
            BiasDirection.LEFT: [
                "progressive",
                "liberal",
                "socialist",
                "leftist",
                "antifa",
                "resistance",
                "social justice",
                "climate action",
            ],
            BiasDirection.RIGHT: [
                "conservative",
                "patriot",
                "traditional",
                "rightist",
                "maga",
                "freedom",
                "liberty",
                "pro-life",
                "second amendment",
            ],
        }

        # Domain type patterns
        self.domain_patterns = {
            SourceType.NEWS_MEDIA: [
                r".*news.*",
                r".*times.*",
                r".*post.*",
                r".*herald.*",
                r".*tribune.*",
                r".*journal.*",
                r".*gazette.*",
            ],
            SourceType.ACADEMIC: [
                r".*\.edu$",
                r".*\.ac\..*",
                r".*university.*",
                r".*college.*",
                r".*research.*",
                r".*institute.*",
            ],
            SourceType.GOVERNMENT: [
                r".*\.gov$",
                r".*\.gov\..*",
                r".*\.mil$",
                r".*\.org$",
            ],
            SourceType.SOCIAL_MEDIA: [
                r"twitter\.com",
                r"facebook\.com",
                r"instagram\.com",
                r"tiktok\.com",
                r"reddit\.com",
                r"youtube\.com",
            ],
            SourceType.WIKI: [r".*wiki.*", r".*pedia.*"],
        }

        # Initialize TF-IDF for content analysis
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2)
        )

        self._load_external_ratings()

    def _load_external_ratings(self):
        """Load external source ratings from various fact-checking organizations."""
        # This would typically load from external APIs or databases
        # For now, we'll use some known ratings
        self.external_ratings = {
            "allsides.com": {  # AllSides media bias ratings
                "cnn.com": {"bias": BiasDirection.CENTER_LEFT, "reliability": 0.75},
                "foxnews.com": {"bias": BiasDirection.RIGHT, "reliability": 0.70},
                "wsj.com": {"bias": BiasDirection.CENTER_RIGHT, "reliability": 0.85},
                "nytimes.com": {"bias": BiasDirection.CENTER_LEFT, "reliability": 0.80},
            },
            "mediabiasfactcheck.com": {
                "reuters.com": {"bias": BiasDirection.CENTER, "reliability": 0.95},
                "apnews.com": {"bias": BiasDirection.CENTER, "reliability": 0.94},
                "breitbart.com": {"bias": BiasDirection.RIGHT, "reliability": 0.45},
            },
        }

    def evaluate_source(
        self, url: str, content: str = None, domain_info: dict[str, Any] = None
    ) -> SourceProfile:
        """
        Comprehensive evaluation of a source.

        Args:
            url: Source URL
            content: Optional content for analysis
            domain_info: Optional domain metadata

        Returns:
            SourceProfile with comprehensive metrics
        """
        domain = self._extract_domain(url)

        # Get or create source profile
        if domain in self.source_profiles:
            profile = self.source_profiles[domain]
        else:
            profile = self._create_new_profile(domain, domain_info)

        # Update metrics
        self._update_source_metrics(profile, url, content, domain_info)

        # Store updated profile
        self.source_profiles[domain] = profile
        profile.updated_at = datetime.now()

        return profile

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except (ValueError, AttributeError):
            return url.lower()

    def _create_new_profile(
        self, domain: str, domain_info: dict[str, Any] = None
    ) -> SourceProfile:
        """Create a new source profile."""
        source_type = self._classify_source_type(domain)
        metrics = SourceMetrics()

        # Initialize with known ratings if available
        if domain in self.reliable_sources:
            metrics.reliability_score = self.reliable_sources[domain]
            metrics.accuracy_score = self.reliable_sources[domain]
        elif domain in self.unreliable_sources:
            metrics.reliability_score = self.unreliable_sources[domain]
            metrics.accuracy_score = self.unreliable_sources[domain]

        profile = SourceProfile(
            domain=domain,
            source_type=source_type,
            metrics=metrics,
            metadata=domain_info or {},
        )

        return profile

    def _classify_source_type(self, domain: str) -> SourceType:
        """Classify source type based on domain patterns."""
        for source_type, patterns in self.domain_patterns.items():
            for pattern in patterns:
                if re.match(pattern, domain, re.IGNORECASE):
                    return source_type
        return SourceType.UNKNOWN

    def _update_source_metrics(
        self,
        profile: SourceProfile,
        url: str,
        content: str = None,
        domain_info: dict[str, Any] = None,
    ):
        """Update source metrics based on new information."""
        metrics = profile.metrics

        # Update external ratings
        self._update_external_ratings(profile)

        # Analyze content if provided
        if content:
            self._analyze_content_quality(profile, content)
            self._analyze_bias_indicators(profile, content)

        # Update transparency score
        self._update_transparency_score(profile, url, domain_info)

        # Update expertise score
        self._update_expertise_score(profile, domain_info)

        # Update recency score
        self._update_recency_score(profile)

        # Calculate overall reliability
        self._calculate_overall_reliability(profile)

    def _update_external_ratings(self, profile: SourceProfile):
        """Update metrics based on external ratings."""
        domain = profile.domain

        for rating_source, ratings in self.external_ratings.items():
            if domain in ratings:
                rating_data = ratings[domain]

                # Update bias information
                if "bias" in rating_data:
                    profile.metrics.bias_direction = rating_data["bias"]

                # Update reliability from external source
                if "reliability" in rating_data:
                    external_reliability = rating_data["reliability"]
                    profile.external_ratings[rating_source] = external_reliability

                    # Weight external ratings
                    if profile.metrics.reliability_score == 0.0:
                        profile.metrics.reliability_score = external_reliability
                    else:
                        # Average with existing score
                        profile.metrics.reliability_score = (
                            profile.metrics.reliability_score + external_reliability
                        ) / 2

    def _analyze_content_quality(self, profile: SourceProfile, content: str):
        """Analyze content quality indicators."""
        metrics = profile.metrics

        # Citation analysis
        citation_indicators = [
            r"\[[\d,\s-]+\]",  # Academic citations [1,2,3]
            r"according to",  # Attribution phrases
            r"study shows",
            r"research indicates",
            r"data from",
            r"source:",
            r"via @",  # Social media attribution
        ]

        citation_count = 0
        for pattern in citation_indicators:
            citation_count += len(re.findall(pattern, content, re.IGNORECASE))

        # Normalize citation score (0-1)
        content_length = len(content.split())
        if content_length > 0:
            citation_density = citation_count / (content_length / 100)  # Per 100 words
            # Cap at 5 citations per 100 words
            metrics.citation_score = min(1.0, citation_density / 5)

        # Transparency indicators
        transparency_indicators = [
            r"correction:",
            r"update:",
            r"editor\'s note",
            r"methodology",
            r"data source",
            r"full disclosure",
        ]

        transparency_count = 0
        for pattern in transparency_indicators:
            transparency_count += len(re.findall(pattern, content, re.IGNORECASE))

        if transparency_count > 0:
            metrics.transparency_score = min(1.0, transparency_count / 3)

    def _analyze_bias_indicators(self, profile: SourceProfile, content: str):
        """Analyze content for bias indicators."""
        content_lower = content.lower()

        bias_scores = {direction: 0 for direction in BiasDirection}

        for direction, keywords in self.bias_keywords.items():
            for keyword in keywords:
                count = content_lower.count(keyword.lower())
                bias_scores[direction] += count

        # Determine dominant bias direction
        max_bias = max(bias_scores.values())
        if max_bias > 0:
            dominant_bias = max(bias_scores, key=bias_scores.get)
            profile.metrics.bias_direction = dominant_bias

            # Calculate bias score (higher = more neutral)
            total_bias_indicators = sum(bias_scores.values())
            content_length = len(content.split())

            if content_length > 0:
                bias_density = total_bias_indicators / (content_length / 100)
                # Invert so higher score = less biased
                profile.metrics.bias_score = max(0.0, 1.0 - (bias_density / 10))

        # Store bias indicators
        for direction, score in bias_scores.items():
            if score > 0:
                profile.bias_indicators.append(f"{direction.value}: {score}")

    def _update_transparency_score(
        self, profile: SourceProfile, url: str, domain_info: dict[str, Any] = None
    ):
        """Update transparency score based on various factors."""
        score = 0.0

        # Check for HTTPS
        if url.startswith("https://"):
            score += 0.2

        # Check for author information
        if domain_info and "author" in domain_info:
            score += 0.3

        # Check for publication date
        if domain_info and "published_date" in domain_info:
            score += 0.2

        # Government and academic sources get transparency bonus
        if profile.source_type in [SourceType.GOVERNMENT, SourceType.ACADEMIC]:
            score += 0.3

        profile.metrics.transparency_score = min(1.0, score)

    def _update_expertise_score(
        self, profile: SourceProfile, domain_info: dict[str, Any] = None
    ):
        """Update expertise score based on source type and domain."""
        score = 0.0

        # Base score by source type
        type_scores = {
            SourceType.ACADEMIC: 0.9,
            SourceType.GOVERNMENT: 0.8,
            SourceType.NEWS_MEDIA: 0.6,
            SourceType.WIKI: 0.5,
            SourceType.BLOG: 0.3,
            SourceType.SOCIAL_MEDIA: 0.2,
            SourceType.FORUM: 0.2,
            SourceType.UNKNOWN: 0.1,
        }

        score = type_scores.get(profile.source_type, 0.1)

        # Adjust based on domain reputation
        if profile.domain in self.reliable_sources:
            score = max(score, 0.8)
        elif profile.domain in self.unreliable_sources:
            score = min(score, 0.3)

        profile.metrics.expertise_score = score

    def _update_recency_score(self, profile: SourceProfile):
        """Update recency score based on last update time."""
        now = datetime.now()
        days_since_update = (now - profile.updated_at).days

        # Exponential decay: score decreases as content gets older
        if days_since_update == 0:
            score = 1.0
        elif days_since_update <= 7:
            score = 0.9
        elif days_since_update <= 30:
            score = 0.7
        elif days_since_update <= 90:
            score = 0.5
        elif days_since_update <= 365:
            score = 0.3
        else:
            score = 0.1

        profile.metrics.recency_score = score

    def _calculate_overall_reliability(self, profile: SourceProfile):
        """Calculate overall reliability score using weighted average."""
        metrics = profile.metrics

        # Weights for different factors
        weights = {
            "accuracy": 0.25,
            "bias": 0.20,
            "transparency": 0.15,
            "expertise": 0.20,
            "recency": 0.10,
            "citation": 0.10,
        }

        # Calculate weighted average
        reliability = (
            weights["accuracy"] * metrics.accuracy_score
            + weights["bias"] * metrics.bias_score
            + weights["transparency"] * metrics.transparency_score
            + weights["expertise"] * metrics.expertise_score
            + weights["recency"] * metrics.recency_score
            + weights["citation"] * metrics.citation_score
        )

        metrics.reliability_score = min(1.0, max(0.0, reliability))

    def update_verification_result(
        self, domain: str, was_correct: bool, claim: str = None, evidence: str = None
    ):
        """Update source metrics based on verification results."""
        if domain not in self.source_profiles:
            # Create basic profile if doesn't exist
            self.source_profiles[domain] = self._create_new_profile(domain)

        profile = self.source_profiles[domain]
        metrics = profile.metrics

        # Update verification counts
        metrics.verification_count += 1
        if was_correct:
            metrics.correct_predictions += 1
        else:
            metrics.false_predictions += 1

        # Update accuracy score
        if metrics.verification_count > 0:
            metrics.accuracy_score = (
                metrics.correct_predictions / metrics.verification_count
            )

        # Store verification history
        verification_record = {
            "timestamp": datetime.now().isoformat(),
            "was_correct": was_correct,
            "claim": claim,
            "evidence": evidence,
        }
        profile.verification_history.append(verification_record)

        # Keep only last 100 records
        if len(profile.verification_history) > 100:
            profile.verification_history = profile.verification_history[-100:]

        # Recalculate overall reliability
        self._calculate_overall_reliability(profile)

        profile.updated_at = datetime.now()

    def get_source_credibility(self, url: str) -> float:
        """Get credibility score for a source URL."""
        domain = self._extract_domain(url)

        if domain in self.source_profiles:
            return self.source_profiles[domain].metrics.reliability_score

        # Return default score for unknown sources
        if domain in self.reliable_sources:
            return self.reliable_sources[domain]
        elif domain in self.unreliable_sources:
            return self.unreliable_sources[domain]

        return 0.5  # Neutral score for unknown sources

    def get_top_sources(self, limit: int = 10) -> list[tuple[str, float]]:
        """Get top sources by reliability score."""
        sources = [
            (domain, profile.metrics.reliability_score)
            for domain, profile in self.source_profiles.items()
        ]
        sources.sort(key=lambda x: x[1], reverse=True)
        return sources[:limit]

    def get_source_analysis(self, domain: str) -> dict[str, Any]:
        """Get detailed analysis for a specific source."""
        if domain not in self.source_profiles:
            return {"error": "Source not found"}

        profile = self.source_profiles[domain]
        metrics = profile.metrics

        return {
            "domain": domain,
            "source_type": profile.source_type.value,
            "overall_reliability": metrics.reliability_score,
            "metrics": {
                "accuracy": metrics.accuracy_score,
                "bias_score": metrics.bias_score,
                "bias_direction": metrics.bias_direction.value,
                "transparency": metrics.transparency_score,
                "expertise": metrics.expertise_score,
                "recency": metrics.recency_score,
                "citation_quality": metrics.citation_score,
            },
            "verification_stats": {
                "total_verifications": metrics.verification_count,
                "correct_predictions": metrics.correct_predictions,
                "false_predictions": metrics.false_predictions,
            },
            "external_ratings": profile.external_ratings,
            "bias_indicators": profile.bias_indicators,
            "last_updated": profile.updated_at.isoformat(),
        }

    def export_profiles(self) -> dict[str, Any]:
        """Export all source profiles for persistence."""
        exported = {}
        for domain, profile in self.source_profiles.items():
            exported[domain] = {
                "domain": profile.domain,
                "source_type": profile.source_type.value,
                "metrics": {
                    "accuracy_score": profile.metrics.accuracy_score,
                    "reliability_score": profile.metrics.reliability_score,
                    "bias_score": profile.metrics.bias_score,
                    "bias_direction": profile.metrics.bias_direction.value,
                    "transparency_score": profile.metrics.transparency_score,
                    "expertise_score": profile.metrics.expertise_score,
                    "recency_score": profile.metrics.recency_score,
                    "citation_score": profile.metrics.citation_score,
                    "verification_count": profile.metrics.verification_count,
                    "correct_predictions": profile.metrics.correct_predictions,
                    "false_predictions": profile.metrics.false_predictions,
                    "last_updated": profile.metrics.last_updated.isoformat(),
                },
                "metadata": profile.metadata,
                "verification_history": profile.verification_history,
                "bias_indicators": profile.bias_indicators,
                "external_ratings": profile.external_ratings,
                "created_at": profile.created_at.isoformat(),
                "updated_at": profile.updated_at.isoformat(),
            }
        return exported

    def import_profiles(self, data: dict[str, Any]):
        """Import source profiles from exported data."""
        for domain, profile_data in data.items():
            metrics = SourceMetrics(
                accuracy_score=profile_data["metrics"]["accuracy_score"],
                reliability_score=profile_data["metrics"]["reliability_score"],
                bias_score=profile_data["metrics"]["bias_score"],
                bias_direction=BiasDirection(profile_data["metrics"]["bias_direction"]),
                transparency_score=profile_data["metrics"]["transparency_score"],
                expertise_score=profile_data["metrics"]["expertise_score"],
                recency_score=profile_data["metrics"]["recency_score"],
                citation_score=profile_data["metrics"]["citation_score"],
                verification_count=profile_data["metrics"]["verification_count"],
                correct_predictions=profile_data["metrics"]["correct_predictions"],
                false_predictions=profile_data["metrics"]["false_predictions"],
                last_updated=datetime.fromisoformat(
                    profile_data["metrics"]["last_updated"]
                ),
            )

            profile = SourceProfile(
                domain=profile_data["domain"],
                source_type=SourceType(profile_data["source_type"]),
                metrics=metrics,
                metadata=profile_data["metadata"],
                verification_history=profile_data["verification_history"],
                bias_indicators=profile_data["bias_indicators"],
                external_ratings=profile_data["external_ratings"],
                created_at=datetime.fromisoformat(profile_data["created_at"]),
                updated_at=datetime.fromisoformat(profile_data["updated_at"]),
            )

            self.source_profiles[domain] = profile
