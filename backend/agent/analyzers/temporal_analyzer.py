"""
Refactored temporal analysis module for detecting temporal mismatches in social media posts.
"""
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dateutil import parser
from dateutil.relativedelta import relativedelta

from agent.analyzers.base_analyzer import BaseAnalyzer
from agent.models.verification_context import VerificationContext

logger = logging.getLogger(__name__)


class TemporalAnalyzer(BaseAnalyzer):
    """Analyzes temporal context of social media posts and claims."""
    
    def __init__(self):
        super().__init__("temporal")
        self.current_time = datetime.now()
        logger.info(f"TemporalAnalyzer initialized with current_time: {self.current_time}")
    
    async def analyze(self, context: VerificationContext) -> Dict[str, Any]:
        """
        Analyze temporal context of a post and its claims.
        
        Args:
            context: Verification context containing all necessary data
            
        Returns:
            Dictionary with temporal analysis results
        """
        extracted_info = context.get_extracted_info()
        
        analysis = {
            "post_timestamp": None,
            "post_age_hours": None,
            "referenced_dates": [],
            "temporal_mismatch": False,
            "mismatch_severity": "none",  # none, minor, major, critical
            "temporal_flags": [],
            "intent_analysis": "unknown"
        }
        
        # Extract post timestamp - prioritize post_date field from image analysis
        post_date_str = extracted_info.get("post_date")
        post_timestamp = self._extract_post_timestamp(post_date_str) if post_date_str else None
        
        # Fallback to extracting from extracted_text if post_date not found
        if not post_timestamp:
            post_timestamp = self._extract_post_timestamp(extracted_info.get("extracted_text", ""))
        
        # Log temporal analysis findings
        if post_timestamp:
            logger.info(f"Extracted post timestamp: {post_timestamp} from {'post_date field' if post_date_str else 'extracted_text'}")
        else:
            logger.warning("No post timestamp found in either post_date field or extracted_text")

        if post_timestamp:
            analysis["post_timestamp"] = post_timestamp
            analysis["post_age_hours"] = self._calculate_age_hours(post_timestamp)
        
        # Extract referenced dates from hierarchical facts
        fact_hierarchy = extracted_info.get("fact_hierarchy", {})
        supporting_facts = fact_hierarchy.get("supporting_facts", [])
        fact_descriptions = [fact.get("description", "") for fact in supporting_facts]
        
        # Also include primary thesis and extracted text for year context
        primary_thesis = fact_hierarchy.get("primary_thesis", "")
        extracted_text = extracted_info.get("extracted_text", "")
        all_text_for_context = [primary_thesis, extracted_text] + fact_descriptions
        
        referenced_dates = self._extract_referenced_dates(fact_descriptions, all_text_for_context)
        analysis["referenced_dates"] = referenced_dates
        
        # Analyze temporal mismatches
        mismatch_analysis = self._analyze_temporal_mismatches(
            post_timestamp, referenced_dates, extracted_info
        )
        analysis.update(mismatch_analysis)
        
        # Analyze potential intent
        analysis["intent_analysis"] = self._analyze_intent(
            post_timestamp, referenced_dates, extracted_info
        )
        
        # Convert datetimes to strings for JSON serialization
        if isinstance(analysis.get("post_timestamp"), datetime):
            analysis["post_timestamp"] = analysis["post_timestamp"].isoformat()

        if "referenced_dates" in analysis:
            analysis["referenced_dates"] = [
                (claim, date.isoformat()) for claim, date in analysis["referenced_dates"]
            ]

        return analysis
    
    def _extract_post_timestamp(self, text: str) -> Optional[datetime]:
        """Extract post timestamp from text."""
        try:
            # Patterns for relative timestamps
            relative_patterns = [
                (r'(\d+)\s*hours?\s*ago', 'hours'),
                (r'(\d+)\s*days?\s*ago', 'days'),
                (r'(\d+)\s*weeks?\s*ago', 'weeks'),
                (r'(\d+)\s*months?\s*ago', 'months'),
                (r'(\d+)\s*years?\s*ago', 'years'),
                (r'(\d+)h\s*ago', 'hours'),
                (r'(\d+)d\s*ago', 'days'),
                (r'(\d+)w\s*ago', 'weeks'),
                (r'(\d+)m\s*ago', 'months'),
                (r'(\d+)y\s*ago', 'years')
            ]
            
            # Try relative timestamps first
            for pattern, unit in relative_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = int(match.group(1))
                    if unit == 'hours':
                        return self.current_time - timedelta(hours=value)
                    elif unit == 'days':
                        return self.current_time - timedelta(days=value)
                    elif unit == 'weeks':
                        return self.current_time - timedelta(weeks=value)
                    elif unit == 'months':
                        return self.current_time - relativedelta(months=value)
                    elif unit == 'years':
                        return self.current_time - relativedelta(years=value)
            
            # Try absolute date patterns
            date_patterns = [
                r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b',  # MM/DD/YYYY or DD/MM/YYYY
                r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',  # YYYY/MM/DD
                r'\b([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\b',  # Month DD, YYYY
                r'\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b',  # DD Month YYYY
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    try:
                        return parser.parse(match)
                    except (ValueError, TypeError, AttributeError) as e:
                        logger.debug(f"Failed to parse date '{match}': {e}")
                        continue
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract post timestamp: {e}")
            return None
    
    def _extract_referenced_dates(self, claims: List[str], context_text: List[str] = None) -> List[Tuple[str, datetime]]:
        """Extract dates referenced in claims."""
        referenced_dates = []
        
        # Look for year context in all available text
        all_context = context_text if context_text else claims
        all_text = " ".join(all_context)
        year_context = None
        year_matches = re.findall(r'\b(20\d{2})\b', all_text)
        if year_matches:
            # Use the most recent explicit year mentioned
            year_context = int(max(year_matches))
            logger.info(f"Found year context: {year_context} in claims")
        
        for claim in claims:
            # Look for dates in claims
            date_patterns = [
                r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b',
                r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
                r'\b([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\b',
                r'\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b',
                r'\b(May\s+\d{1,2})\b',  # Specific for the example
                r'\b([A-Za-z]{3,9}\s+\d{1,2})\b',  # General month/day pattern
                r'\b(\d{4})\b'  # Just year
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, claim)
                for match in matches:
                    try:
                        # Parse the date
                        parsed_date = parser.parse(match)
                        
                        # If only month/day is provided without year, use year context or intelligent defaults
                        if len(match.split()) == 2 and not re.search(r'\d{4}', match):  # e.g., "May 21" without year
                            if year_context:
                                # Use the year context found in the content
                                parsed_date = parsed_date.replace(year=year_context)
                                logger.info(f"Applied year context {year_context} to '{match}' -> {parsed_date}")
                            else:
                                # Fallback: if the date would be in the future with current year, assume previous year
                                current_date = self.current_time
                                if parsed_date > current_date:
                                    parsed_date = parsed_date.replace(year=current_date.year - 1)
                                    logger.info(f"Applied previous year logic to '{match}' -> {parsed_date}")
                        
                        referenced_dates.append((claim, parsed_date))
                        logger.info(f"Parsed referenced date: '{match}' -> {parsed_date} in claim: {claim}")
                    except (ValueError, TypeError, AttributeError) as e:
                        logger.debug(f"Failed to parse referenced date '{match}' in claim '{claim[:50]}...': {e}")
                        continue
        
        return referenced_dates
    
    def _calculate_age_hours(self, timestamp: datetime) -> float:
        """Calculate age of post in hours."""
        return (self.current_time - timestamp).total_seconds() / 3600
    
    def _analyze_temporal_mismatches(
        self, 
        post_timestamp: Optional[datetime], 
        referenced_dates: List[Tuple[str, datetime]],
        extracted_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze temporal mismatches between post time and referenced events."""
        analysis = {
            "temporal_mismatch": False,
            "mismatch_severity": "none",
            "temporal_flags": []
        }
        
        if not post_timestamp or not referenced_dates:
            return analysis
        
        for claim, ref_date in referenced_dates:
            # Calculate time difference
            time_diff = post_timestamp - ref_date
            days_diff = abs(time_diff.days)
            
            # Check for significant temporal mismatches
            if days_diff > 365:  # More than a year
                analysis["temporal_mismatch"] = True
                analysis["mismatch_severity"] = "critical"
                analysis["temporal_flags"].append(
                    f"Post references event from {days_diff} days ago: {claim}"
                )
            elif days_diff > 90:  # More than 3 months
                analysis["temporal_mismatch"] = True
                if analysis["mismatch_severity"] != "critical":
                    analysis["mismatch_severity"] = "major"
                analysis["temporal_flags"].append(
                    f"Post references event from {days_diff} days ago: {claim}"
                )
            elif days_diff > 30:  # More than a month
                analysis["temporal_mismatch"] = True
                if analysis["mismatch_severity"] == "none":
                    analysis["mismatch_severity"] = "minor"
                analysis["temporal_flags"].append(
                    f"Post references event from {days_diff} days ago: {claim}"
                )
        
        return analysis
    
    def _analyze_intent(
        self, 
        post_timestamp: Optional[datetime], 
        referenced_dates: List[Tuple[str, datetime]],
        extracted_info: Dict[str, Any]
    ) -> str:
        """Analyze potential intent behind temporal mismatches."""
        if not post_timestamp or not referenced_dates:
            return "unknown"
        
        # Check for financial/crypto content with temporal mismatches
        content = extracted_info.get("extracted_text", "").lower()
        financial_keywords = ['bitcoin', 'btc', 'crypto', 'investment', 'buy', 'sell', 'price', 'market', 'blackrock']
        
        is_financial = any(keyword in content for keyword in financial_keywords)
        
        # Check for different levels of temporal mismatch
        max_days_diff = 0
        for _, ref_date in referenced_dates:
            days_diff = abs((post_timestamp - ref_date).days)
            max_days_diff = max(max_days_diff, days_diff)
        
        # Log the mismatch for debugging
        logger.info(f"Temporal mismatch analysis: max_days_diff={max_days_diff}, is_financial={is_financial}")
        logger.info(f"Post timestamp: {post_timestamp}, Referenced dates: {[(claim[:50], date) for claim, date in referenced_dates]}")
        
        if is_financial:
            if max_days_diff > 365:  # More than a year
                return "potentially_misleading_old_news"
            elif max_days_diff > 180:  # More than 6 months
                # Look for manipulation indicators
                manipulation_keywords = ['pump', 'moon', 'buy now', 'urgent', 'breaking', 'biggest', 'massive']
                if any(keyword in content for keyword in manipulation_keywords):
                    return "potential_market_manipulation"
                else:
                    return "outdated_financial_information"
            elif max_days_diff > 30:  # More than a month
                return "outdated_financial_information"
            else:
                return "legitimate_recent_content"
        elif max_days_diff > 90:  # Non-financial content more than 3 months old
            return "outdated_information_sharing"
        else:
            return "legitimate_recent_content" 