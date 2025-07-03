"""
Motives analysis module for analyzing potential motives behind social media posts.
"""
import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from agent.llm import llm_manager
from agent.prompts import MANIPULATION_DETECTION_PROMPT, ADVERSARIAL_ROBUSTNESS_PROMPT

logger = logging.getLogger(__name__)


class MotivesAnalyzer:
    """Analyzes potential motives behind social media posts."""
    
    def __init__(self):
        self.financial_manipulation_keywords = [
            'pump', 'moon', 'rocket', 'buy now', 'urgent', 'breaking',
            'massive gains', 'last chance', 'don\'t miss', 'explosive growth',
            'instant profits', 'guaranteed returns', 'insider info', 'secret'
        ]
        
        self.misinformation_indicators = [
            'mainstream media won\'t tell you', 'they don\'t want you to know',
            'hidden truth', 'wake up', 'do your own research', 'sheeple',
            'government lies', 'big pharma', 'conspiracy', 'cover-up'
        ]
        
        self.emotional_manipulation_keywords = [
            'shocking', 'unbelievable', 'outrageous', 'scandal', 'exposed',
            'you won\'t believe', 'this will blow your mind', 'incredible',
            'absolutely insane', 'mind-blowing'
        ]
        
        self.engagement_farming_patterns = [
            'like if you agree', 'share if you care', 'comment below',
            'tag someone', 'retweet if', 'follow for more', 'turn on notifications'
        ]
    
    async def analyze_motives(self, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze potential motives behind a social media post.
        
        Args:
            extracted_info: Dictionary containing extracted post information
            
        Returns:
            Dictionary with motives analysis results
        """
        try:
            content = extracted_info.get("extracted_text", "").lower()
            claims = extracted_info.get("claims", [])
            primary_topic = extracted_info.get("primary_topic", "general")
            temporal_analysis = extracted_info.get("temporal_analysis", {})
            
            motives_analysis = {
                "primary_motive": "unknown",
                "confidence_score": 0.0,
                "secondary_motives": [],
                "manipulation_indicators": [],
                "credibility_assessment": "neutral",
                "risk_level": "low",
                "detailed_analysis": {}
            }
            
            # Analyze different types of motives
            financial_analysis = self._analyze_financial_motives(content, claims, temporal_analysis)
            misinformation_analysis = self._analyze_misinformation_motives(content, claims)
            engagement_analysis = self._analyze_engagement_motives(content)
            emotional_analysis = self._analyze_emotional_manipulation(content)
            
            # Advanced LLM-based manipulation detection
            llm_manipulation_analysis = await self._analyze_llm_manipulation(content, claims, temporal_analysis)
            
            # Combine analyses
            all_analyses = {
                "financial": financial_analysis,
                "misinformation": misinformation_analysis,
                "engagement": engagement_analysis,
                "emotional": emotional_analysis,
                "llm_manipulation": llm_manipulation_analysis
            }
            
            # Determine primary motive
            primary_motive, confidence = self._determine_primary_motive(all_analyses)
            motives_analysis["primary_motive"] = primary_motive
            motives_analysis["confidence_score"] = confidence
            
            # Collect secondary motives
            secondary_motives = []
            for motive_type, analysis in all_analyses.items():
                if motive_type != primary_motive and analysis["confidence"] > 0.3:
                    secondary_motives.append({
                        "type": motive_type,
                        "confidence": analysis["confidence"]
                    })
            motives_analysis["secondary_motives"] = secondary_motives
            
            # Collect manipulation indicators
            manipulation_indicators = []
            for analysis in all_analyses.values():
                manipulation_indicators.extend(analysis.get("indicators", []))
            motives_analysis["manipulation_indicators"] = list(set(manipulation_indicators))
            
            # Assess overall credibility and risk
            motives_analysis["credibility_assessment"] = self._assess_credibility(all_analyses)
            motives_analysis["risk_level"] = self._assess_risk_level(all_analyses)
            
            # Store detailed analysis
            motives_analysis["detailed_analysis"] = all_analyses
            
            logger.info(f"Motives analysis completed: {primary_motive} (confidence: {confidence:.2f})")
            return motives_analysis
            
        except Exception as e:
            logger.error(f"Motives analysis failed: {e}")
            return {
                "primary_motive": "unknown",
                "confidence_score": 0.0,
                "secondary_motives": [],
                "manipulation_indicators": [f"Analysis failed: {e}"],
                "credibility_assessment": "neutral",
                "risk_level": "low",
                "detailed_analysis": {}
            }
    
    def _analyze_financial_motives(self, content: str, claims: List[str], temporal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential financial manipulation motives."""
        manipulation_score = 0.0
        indicators = []
        
        # Check for financial manipulation keywords
        for keyword in self.financial_manipulation_keywords:
            if keyword in content:
                manipulation_score += 0.15
                indicators.append(f"Financial manipulation keyword: '{keyword}'")
        
        # Check for urgency indicators
        urgency_patterns = [
            r'act now', r'limited time', r'expires soon', r'hurry',
            r'only \d+ left', r'final hours', r'last chance'
        ]
        for pattern in urgency_patterns:
            if re.search(pattern, content):
                manipulation_score += 0.2
                indicators.append(f"Urgency indicator: '{pattern}'")
        
        # Check temporal context for potential manipulation
        intent_analysis = temporal_analysis.get("intent_analysis", "")
        if intent_analysis == "potential_market_manipulation":
            manipulation_score += 0.4
            indicators.append("Temporal mismatch suggests potential market manipulation")
        elif intent_analysis == "outdated_financial_information":
            manipulation_score += 0.2
            indicators.append("Outdated financial information being shared")
        
        # Check for specific financial claims without context
        financial_claims = [claim for claim in claims if any(
            keyword in claim.lower() for keyword in ['bitcoin', 'btc', 'crypto', 'stock', 'investment', 'buy', 'sell']
        )]
        if financial_claims and manipulation_score > 0.3:
            manipulation_score += 0.2
            indicators.append("Financial claims combined with manipulation indicators")
        
        return {
            "confidence": min(manipulation_score, 1.0),
            "indicators": indicators,
            "assessment": "high_risk" if manipulation_score > 0.6 else "moderate_risk" if manipulation_score > 0.3 else "low_risk"
        }
    
    def _analyze_misinformation_motives(self, content: str, claims: List[str]) -> Dict[str, Any]:
        """Analyze potential misinformation motives."""
        misinformation_score = 0.0
        indicators = []
        
        # Check for misinformation keywords
        for keyword in self.misinformation_indicators:
            if keyword in content:
                misinformation_score += 0.2
                indicators.append(f"Misinformation indicator: '{keyword}'")
        
        # Check for anti-establishment rhetoric
        anti_establishment_patterns = [
            r'they don\'t want you to know', r'hidden agenda', r'truth they hide',
            r'media manipulation', r'fake news', r'propaganda'
        ]
        for pattern in anti_establishment_patterns:
            if re.search(pattern, content):
                misinformation_score += 0.25
                indicators.append(f"Anti-establishment rhetoric: '{pattern}'")
        
        # Check for unsupported claims
        absolute_claims = [
            r'always', r'never', r'all', r'none', r'every', r'completely',
            r'totally', r'absolutely', r'guaranteed', r'proven fact'
        ]
        for pattern in absolute_claims:
            if re.search(pattern, content) and len(claims) > 0:
                misinformation_score += 0.1
                indicators.append(f"Absolute claim without evidence: '{pattern}'")
        
        return {
            "confidence": min(misinformation_score, 1.0),
            "indicators": indicators,
            "assessment": "high_risk" if misinformation_score > 0.6 else "moderate_risk" if misinformation_score > 0.3 else "low_risk"
        }
    
    def _analyze_engagement_motives(self, content: str) -> Dict[str, Any]:
        """Analyze potential engagement farming motives."""
        engagement_score = 0.0
        indicators = []
        
        # Check for engagement farming patterns
        for pattern in self.engagement_farming_patterns:
            if pattern in content:
                engagement_score += 0.2
                indicators.append(f"Engagement farming: '{pattern}'")
        
        # Check for clickbait elements
        clickbait_patterns = [
            r'you won\'t believe', r'this will shock you', r'amazing trick',
            r'doctors hate this', r'one weird trick', r'this changes everything'
        ]
        for pattern in clickbait_patterns:
            if re.search(pattern, content):
                engagement_score += 0.15
                indicators.append(f"Clickbait element: '{pattern}'")
        
        return {
            "confidence": min(engagement_score, 1.0),
            "indicators": indicators,
            "assessment": "high_engagement_focus" if engagement_score > 0.5 else "moderate_engagement_focus" if engagement_score > 0.2 else "low_engagement_focus"
        }
    
    def _analyze_emotional_manipulation(self, content: str) -> Dict[str, Any]:
        """Analyze potential emotional manipulation tactics."""
        emotional_score = 0.0
        indicators = []
        
        # Check for emotional manipulation keywords
        for keyword in self.emotional_manipulation_keywords:
            if keyword in content:
                emotional_score += 0.15
                indicators.append(f"Emotional manipulation: '{keyword}'")
        
        # Check for fear-based content
        fear_patterns = [
            r'dangerous', r'scary', r'terrifying', r'disaster', r'crisis',
            r'emergency', r'threat', r'warning', r'alert', r'beware'
        ]
        for pattern in fear_patterns:
            if re.search(pattern, content):
                emotional_score += 0.2
                indicators.append(f"Fear-based content: '{pattern}'")
        
        # Check for FOMO (Fear of Missing Out)
        fomo_patterns = [
            r'don\'t miss out', r'limited time', r'exclusive', r'special offer',
            r'act fast', r'while supplies last', r'everyone is doing it'
        ]
        for pattern in fomo_patterns:
            if re.search(pattern, content):
                emotional_score += 0.18
                indicators.append(f"FOMO tactic: '{pattern}'")
        
        return {
            "confidence": min(emotional_score, 1.0),
            "indicators": indicators,
            "assessment": "high_emotional_manipulation" if emotional_score > 0.6 else "moderate_emotional_manipulation" if emotional_score > 0.3 else "low_emotional_manipulation"
        }
    
    def _determine_primary_motive(self, analyses: Dict[str, Dict[str, Any]]) -> tuple:
        """Determine the primary motive based on all analyses."""
        max_confidence = 0.0
        primary_motive = "legitimate_sharing"
        
        for motive_type, analysis in analyses.items():
            confidence = analysis["confidence"]
            if confidence > max_confidence:
                max_confidence = confidence
                primary_motive = motive_type
        
        # If no significant motive detected, assume legitimate sharing
        if max_confidence < 0.3:
            primary_motive = "legitimate_sharing"
            max_confidence = 0.7  # Moderate confidence in legitimate sharing
        
        return primary_motive, max_confidence
    
    def _assess_credibility(self, analyses: Dict[str, Dict[str, Any]]) -> str:
        """Assess overall credibility of the post."""
        # Weight LLM manipulation analysis higher due to its semantic understanding
        llm_confidence = analyses.get("llm_manipulation", {}).get("confidence", 0)
        other_analyses = [analysis for key, analysis in analyses.items() if key != "llm_manipulation"]
        
        if other_analyses:
            traditional_score = sum(analysis["confidence"] for analysis in other_analyses) / len(other_analyses)
            # Weighted average: LLM analysis gets 60% weight, traditional methods get 40%
            manipulation_score = (llm_confidence * 0.6) + (traditional_score * 0.4)
        else:
            manipulation_score = llm_confidence
        
        if manipulation_score > 0.7:
            return "low_credibility"
        elif manipulation_score > 0.4:
            return "questionable_credibility"
        elif manipulation_score > 0.2:
            return "moderate_credibility"
        else:
            return "high_credibility"
    
    def _assess_risk_level(self, analyses: Dict[str, Dict[str, Any]]) -> str:
        """Assess overall risk level of the post."""
        # LLM-detected manipulation poses highest risk
        if analyses.get("llm_manipulation", {}).get("confidence", 0) > 0.7:
            return "high"
        
        # Financial manipulation poses highest risk
        if analyses["financial"]["confidence"] > 0.6:
            return "high"
        
        # Misinformation poses significant risk
        if analyses["misinformation"]["confidence"] > 0.6:
            return "high"
        
        # Moderate risk for LLM-detected manipulation
        if analyses.get("llm_manipulation", {}).get("confidence", 0) > 0.4:
            return "moderate"
        
        # Moderate risk for emotional manipulation
        if analyses["emotional"]["confidence"] > 0.5:
            return "moderate"
        
        # Low risk for engagement farming only
        if analyses["engagement"]["confidence"] > 0.3:
            return "low"
        
        return "low"
    
    async def _analyze_llm_manipulation(self, content: str, claims: List[str], temporal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to detect sophisticated manipulation techniques."""
        try:
            logger.info("Starting LLM-based manipulation analysis")
            
            # First, adversarial robustness check
            adversarial_prompt = ADVERSARIAL_ROBUSTNESS_PROMPT.format_messages(content=content)
            adversarial_full_prompt = adversarial_prompt[0].content + "\n\n" + adversarial_prompt[1].content
            logger.debug(f"Adversarial check prompt: {adversarial_full_prompt}")
            
            adversarial_check_content = await llm_manager.invoke_text_only(text=adversarial_full_prompt)
            logger.info(f"Adversarial check result: {adversarial_check_content}")
            
            # Main manipulation detection
            manipulation_prompt = MANIPULATION_DETECTION_PROMPT.format_messages(
                content=content,
                claims=str(claims),
                context="Social media post analysis",
                temporal_analysis=str(temporal_analysis)
            )
            manipulation_full_prompt = manipulation_prompt[0].content + "\n\n" + manipulation_prompt[1].content
            logger.debug(f"Manipulation detection prompt: {manipulation_full_prompt}")
            
            manipulation_analysis_content = await llm_manager.invoke_text_only(text=manipulation_full_prompt)
            logger.info(f"Manipulation analysis raw response length: {len(manipulation_analysis_content)} chars")
            
            # Parse LLM response
            try:
                analysis_text = manipulation_analysis_content
                logger.debug(f"Raw LLM analysis response: {analysis_text}")
                
                # Try multiple JSON extraction methods
                parsed_analysis = None
                
                # Method 1: Try to find complete JSON object
                import re
                json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', analysis_text, re.DOTALL)
                
                for json_match in json_matches:
                    try:
                        parsed_analysis = json.loads(json_match)
                        logger.debug(f"Successfully parsed JSON: {parsed_analysis}")
                        break
                    except json.JSONDecodeError:
                        continue
                
                # Method 2: Try to extract from code blocks
                if not parsed_analysis:
                    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', analysis_text, re.DOTALL)
                    if code_block_match:
                        try:
                            parsed_analysis = json.loads(code_block_match.group(1))
                            logger.debug(f"Parsed JSON from code block: {parsed_analysis}")
                        except json.JSONDecodeError:
                            pass
                
                # Method 3: Try to parse the entire response as JSON
                if not parsed_analysis:
                    try:
                        parsed_analysis = json.loads(analysis_text.strip())
                        logger.debug(f"Parsed entire response as JSON: {parsed_analysis}")
                    except json.JSONDecodeError:
                        pass
                
                # Fallback: Create structure from text analysis
                if not parsed_analysis:
                    logger.warning(f"All JSON parsing methods failed, creating fallback analysis")
                    
                    # Basic text analysis for manipulation indicators
                    manipulation_keywords = ['manipulation', 'misleading', 'deceptive', 'false', 'propaganda']
                    detected_manipulations = [keyword for keyword in manipulation_keywords if keyword.lower() in analysis_text.lower()]
                    
                    confidence = min(len(detected_manipulations) * 0.2, 0.8)
                    
                    parsed_analysis = {
                        "manipulation_detected": len(detected_manipulations) > 0,
                        "manipulation_types": detected_manipulations,
                        "confidence": confidence,
                        "reasoning": f"Fallback analysis detected: {', '.join(detected_manipulations) if detected_manipulations else 'No clear manipulation indicators'}"
                    }
                
                # Extract confidence and indicators
                confidence = parsed_analysis.get("confidence", 0.0)
                if isinstance(confidence, str):
                    confidence = float(confidence.rstrip('%')) / 100 if '%' in confidence else 0.0
                
                logger.info(f"Successfully parsed manipulation analysis: manipulation_detected={parsed_analysis.get('manipulation_detected', False)}, confidence={confidence}")
                
                indicators = []
                if parsed_analysis.get("manipulation_detected", False):
                    indicators.extend(parsed_analysis.get("manipulation_types", []))
                    if "reasoning" in parsed_analysis:
                        indicators.append(f"LLM Analysis: {parsed_analysis['reasoning']}")
                
                # Check for adversarial content
                if "manipulation detected" in adversarial_check_content.lower():
                    indicators.append(f"Adversarial content detected: {adversarial_check_content}")
                    confidence = max(confidence, 0.8)
                elif "safe" not in adversarial_check_content.lower():
                    indicators.append("Uncertain adversarial check result")
                    confidence = max(confidence, 0.3)
                
                return {
                    "confidence": confidence,
                    "indicators": indicators,
                    "assessment": "high_risk" if confidence > 0.7 else "moderate_risk" if confidence > 0.4 else "low_risk",
                    "llm_analysis": parsed_analysis,
                    "adversarial_check": adversarial_check_content
                }
                
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"Failed to parse LLM manipulation analysis: {e}")
                return {
                    "confidence": 0.0,
                    "indicators": ["LLM analysis parsing failed"],
                    "assessment": "low_risk",
                    "llm_analysis": {"error": str(e)},
                    "adversarial_check": adversarial_check_content if 'adversarial_check_content' in locals() else ""
                }
                
        except Exception as e:
            logger.error(f"LLM manipulation analysis failed: {e}")
            return {
                "confidence": 0.0,
                "indicators": [f"LLM analysis failed: {e}"],
                "assessment": "low_risk",
                "llm_analysis": {"error": str(e)},
                "adversarial_check": ""
            }


# Global motives analyzer instance
motives_analyzer = MotivesAnalyzer() 