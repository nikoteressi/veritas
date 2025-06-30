"""
Temporal analysis module for detecting temporal mismatches in social media posts.
"""
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dateutil import parser
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """Analyzes temporal context of social media posts and claims."""
    
    def __init__(self):
        self.current_time = datetime.now()
    
    def analyze_temporal_context(self, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze temporal context of a post and its claims.
        
        Args:
            extracted_info: Dictionary containing extracted post information
            
        Returns:
            Dictionary with temporal analysis results
        """
        try:
            analysis = {
                "post_timestamp": None,
                "post_age_hours": None,
                "referenced_dates": [],
                "temporal_mismatch": False,
                "mismatch_severity": "none",  # none, minor, major, critical
                "temporal_flags": [],
                "intent_analysis": "unknown"
            }
            
            # Extract post timestamp
            post_date_str = extracted_info.get("post_date")
            post_timestamp = self._extract_post_timestamp(post_date_str) if post_date_str else None
            
            if not post_timestamp:
                post_timestamp = self._extract_post_timestamp(extracted_info.get("extracted_text", ""))

            if post_timestamp:
                analysis["post_timestamp"] = post_timestamp
                analysis["post_age_hours"] = self._calculate_age_hours(post_timestamp)
            
            # Extract referenced dates from claims
            referenced_dates = self._extract_referenced_dates(extracted_info.get("claims", []))
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

            logger.info(f"Temporal analysis completed: {analysis['mismatch_severity']} mismatch detected")
            return analysis
            
        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            return {
                "post_timestamp": None,
                "post_age_hours": None,
                "referenced_dates": [],
                "temporal_mismatch": False,
                "mismatch_severity": "none",
                "temporal_flags": [f"Analysis failed: {e}"],
                "intent_analysis": "unknown"
            }
    
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
                    except:
                        continue
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract post timestamp: {e}")
            return None
    
    def _extract_referenced_dates(self, claims: List[str]) -> List[Tuple[str, datetime]]:
        """Extract dates referenced in claims."""
        referenced_dates = []
        
        for claim in claims:
            # Look for dates in claims
            date_patterns = [
                r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b',
                r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
                r'\b([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\b',
                r'\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b',
                r'\b(May\s+\d{1,2})\b',  # Specific for the example
                r'\b(\d{4})\b'  # Just year
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, claim)
                for match in matches:
                    try:
                        parsed_date = parser.parse(match)
                        referenced_dates.append((claim, parsed_date))
                    except:
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
        financial_keywords = ['bitcoin', 'btc', 'crypto', 'investment', 'buy', 'sell', 'price', 'market']
        
        is_financial = any(keyword in content for keyword in financial_keywords)
        
        # Check for significant temporal mismatch
        has_major_mismatch = any(
            abs((post_timestamp - ref_date).days) > 90 
            for _, ref_date in referenced_dates
        )
        
        if is_financial and has_major_mismatch:
            # Look for manipulation indicators
            manipulation_keywords = ['pump', 'moon', 'buy now', 'urgent', 'breaking']
            if any(keyword in content for keyword in manipulation_keywords):
                return "potential_market_manipulation"
            else:
                return "outdated_financial_information"
        elif has_major_mismatch:
            return "outdated_information_sharing"
        else:
            return "legitimate_recent_content"


# Global temporal analyzer instance
temporal_analyzer = TemporalAnalyzer()
