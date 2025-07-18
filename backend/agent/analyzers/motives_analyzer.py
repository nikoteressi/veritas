"""
Refactored motives analysis module for analyzing potential motives behind social media posts.
Now uses fact-check verdicts and temporal analysis as primary inputs for informed motive determination.
"""
import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser

from agent.analyzers.base_analyzer import BaseAnalyzer
from agent.models.verification_context import VerificationContext

from app.exceptions import MotivesAnalysisError

logger = logging.getLogger(__name__)


class MotiveAnalysisResult(BaseModel):
    primary_motive: str = Field(..., description="The most likely motive behind the post.")
    confidence_score: float = Field(..., description="Confidence in the motive analysis, from 0.0 to 1.0.")
    reasoning: str = Field(..., description="Explanation for the identified motive.")
    risk_level: str = Field(..., description="Estimated risk level (e.g., low, moderate, high).")
    manipulation_indicators: List[str] = Field(default_factory=list, description="Indicators of manipulation detected.")


class MotivesAnalyzer(BaseAnalyzer):
    """Analyzes potential motives behind social media posts using an LLM."""

    def __init__(self, llm_manager, prompt_manager):
        super().__init__("motives")
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager

    async def analyze(self, context: VerificationContext) -> Dict[str, Any]:
        """Analyzes motives using the reasoning LLM and context data."""
        if not all([context.screenshot_data, context.temporal_analysis, context.verdict_result]):
            raise MotivesAnalysisError("Missing required data in context for motives analysis.")

        output_parser = PydanticOutputParser(pydantic_object=MotiveAnalysisResult)

        try:
            prompt_template = self.prompt_manager.get_prompt_template("motives_analysis_llm")
            prompt = await prompt_template.aformat(final_verdict=context.verdict_result.verdict,
                                             verdict_confidence=context.verdict_result.confidence_score,
                                             temporal_analysis=context.temporal_analysis,
                                             primary_topic=context.primary_topic,
                                             screenshot_data=context.screenshot_data,
                                             format_instructions=output_parser.get_format_instructions())

            logger.info(f"Formatted prompt for LLM: {prompt}")

            response = await self.llm_manager.invoke_text_only(prompt)
            parsed_response = output_parser.parse(response)

            if not isinstance(parsed_response, MotiveAnalysisResult):
                raise MotivesAnalysisError("Failed to parse motive analysis from LLM response.")

            logger.info("Successfully performed LLM-based motives analysis.")
            return parsed_response

        except Exception as e:
            logger.error(f"Error during LLM-based motives analysis: {e}", exc_info=True)
            raise MotivesAnalysisError(f"An unexpected error occurred during motives analysis: {e}") from e
    
    def _extract_fact_check_verdict(self, fact_check_result) -> str:
        """Extract the fact-check verdict from the result."""
        if hasattr(fact_check_result, 'claim_results') and fact_check_result.claim_results:
            # Look for overall assessment in claim results
            for claim_result in fact_check_result.claim_results:
                if 'assessment' in claim_result:
                    logger.info(f"Fact-check verdict: {claim_result['summary']}")
                    return claim_result['summary']
        
        # Look for verdict in summary
        if hasattr(fact_check_result, 'summary'):
            summary = fact_check_result.summary
            if hasattr(summary, 'dict'):
                summary_dict = summary.dict()
                for key, value in summary_dict.items():
                    if 'assessment' in key.lower() or 'verdict' in key.lower():
                        logger.info(f"Fact-check verdict2: {value}")
                        return str(value)
        
        return "unknown"
    
    def _extract_fact_check_confidence(self, fact_check_result) -> float:
        """Extract confidence score from fact-check results."""
        if hasattr(fact_check_result, 'claim_results') and fact_check_result.claim_results:
            confidences = []
            for claim_result in fact_check_result.claim_results:
                if 'confidence' in claim_result:
                    confidences.append(claim_result['confidence'])
            
            if confidences:
                return sum(confidences) / len(confidences) / 100.0  # Normalize to 0-1
        
        return 0.5  # Default moderate confidence
    
    async def _analyze_motives_with_llm(
        self,
        content: str,
        fact_check_verdict: str,
        fact_check_confidence: float,
        temporal_analysis: Dict[str, Any],
        primary_topic: str
    ) -> Dict[str, Any]:
        """Use LLM to analyze motives based on fact-check results."""
        
        # Format temporal analysis for readability
        temporal_summary = self._format_temporal_analysis(temporal_analysis)
        
        # Create prompt with fact-check context
        prompt_template = prompt_manager.get_prompt_template("motives_analysis")
        prompt = await prompt_template.aformat(
            fact_check_verdict=fact_check_verdict,
            fact_check_confidence=fact_check_confidence,
            temporal_analysis=temporal_summary,
            primary_topic=primary_topic,
            content=content
        )
        logger.info(f"Motives analysis prompt: {prompt}")
        
        # Get LLM response
        response = await llm_manager.invoke_text_only(prompt)
        
        # Parse JSON response
        try:
            # Clean response to extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                return {
                    "primary_motive": result.get("primary_motive", "unknown"),
                    "confidence_score": result.get("confidence_score", 0.5),
                    "reasoning": result.get("reasoning", "LLM analysis completed"),
                    "risk_level": result.get("risk_level", "moderate"),
                    "manipulation_indicators": result.get("manipulation_indicators", []),
                    "fact_check_informed": True,
                    "analysis_method": "fact_check_informed_llm"
                }
        except Exception as parse_error:
            logger.warning(f"Failed to parse LLM motives analysis: {parse_error}")
        
        # Fallback to rule-based analysis if LLM parsing fails
        return self._rule_based_analysis(fact_check_verdict, primary_topic, temporal_analysis)
    
    async def _analyze_motives_with_verdict(
        self,
        content: str,
        final_verdict: str,
        verdict_confidence: float,
        verdict_reasoning: str,
        temporal_analysis: Dict[str, Any],
        primary_topic: str,
        post_date: str,
        mentioned_dates: list
    ) -> Dict[str, Any]:
        """Use LLM to analyze motives based on final verdict."""
        
        # Format temporal analysis for readability
        temporal_summary = self._format_temporal_analysis(temporal_analysis)
        
        # Format dates information
        dates_info = f"Post Date: {post_date}, Mentioned Dates: {', '.join(mentioned_dates) if mentioned_dates else 'None'}"
        
        # Create prompt with verdict context
        prompt_template = prompt_manager.get_prompt_template("motives_analysis")
        prompt = await prompt_template.aformat(
            fact_check_verdict=f"{final_verdict} (Confidence: {verdict_confidence:.2f})",
            fact_check_confidence=verdict_confidence,
            temporal_analysis=f"{temporal_summary} | {dates_info}",
            primary_topic=primary_topic,
            content=content
        )
        
        # Add verdict reasoning as additional context
        full_prompt = f"{prompt}\n\n**Verdict Reasoning**: {verdict_reasoning}"
        
        logger.info(f"Motives analysis prompt: {full_prompt}")
        
        # Get LLM response
        response = await llm_manager.invoke_text_only(full_prompt)
        
        # Parse JSON response
        try:
            # Clean response to extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                return {
                    "primary_motive": result.get("primary_motive", "unknown"),
                    "confidence_score": result.get("confidence_score", 0.5),
                    "reasoning": result.get("reasoning", "LLM analysis completed"),
                    "risk_level": result.get("risk_level", "moderate"),
                    "manipulation_indicators": result.get("manipulation_indicators", []),
                    "verdict_informed": True,
                    "analysis_method": "verdict_informed_llm"
                }
        except Exception as parse_error:
            logger.warning(f"Failed to parse LLM motives analysis: {parse_error}")
        
        # Fallback to rule-based analysis if LLM parsing fails
        return self._rule_based_analysis_with_verdict(final_verdict, primary_topic, temporal_analysis, post_date, mentioned_dates)
    
    def _format_temporal_analysis(self, temporal_analysis: Dict[str, Any]) -> str:
        """Format temporal analysis for prompt readability."""
        if not temporal_analysis:
            return "No temporal analysis available"
        
        # Extract key information
        timing_assessment = temporal_analysis.get("timing_assessment", "unknown")
        intent_analysis = temporal_analysis.get("intent_analysis", "unknown")
        temporal_verdict = temporal_analysis.get("temporal_verdict", "unknown")
        
        return f"Timing: {timing_assessment}, Intent: {intent_analysis}, Verdict: {temporal_verdict}"
    
    def _rule_based_analysis(
        self, 
        fact_check_verdict: str, 
        primary_topic: str, 
        temporal_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback rule-based analysis using fact-check context."""
        
        # Determine motive based on fact-check verdict
        if fact_check_verdict.lower() in ["false", "likely_false"]:
            if primary_topic == "financial":
                primary_motive = "financial_manipulation"
                risk_level = "high"
                reasoning = "Content contains false financial information, likely intended to manipulate markets"
            else:
                primary_motive = "disinformation"
                risk_level = "high"
                reasoning = "Content contains false information, likely intended to mislead"
        
        elif fact_check_verdict.lower() in ["true", "likely_true"]:
            # Check temporal context
            temporal_verdict = temporal_analysis.get("temporal_verdict", "appropriate")
            if temporal_verdict == "misleading_timing":
                primary_motive = "outdated_information"
                risk_level = "moderate"
                reasoning = "Content is factually true but presented as current when it's outdated"
            else:
                primary_motive = "informing"
                risk_level = "low"
                reasoning = "Content appears to be genuinely informative with true information"
        
        else:  # unverified, partially true, etc.
            primary_motive = "unknown"
            risk_level = "moderate"
            reasoning = "Motive unclear due to mixed or unverified fact-check results"
        
        return {
            "primary_motive": primary_motive,
            "confidence_score": 0.7,
            "reasoning": reasoning,
            "risk_level": risk_level,
            "manipulation_indicators": self._identify_manipulation_indicators(fact_check_verdict, primary_topic),
            "fact_check_informed": True,
            "analysis_method": "rule_based_with_fact_check"
        }
    
    def _rule_based_analysis_with_verdict(
        self, 
        final_verdict: str, 
        primary_topic: str, 
        temporal_analysis: Dict[str, Any],
        post_date: str,
        mentioned_dates: list
    ) -> Dict[str, Any]:
        """Rule-based analysis using final verdict and temporal context."""
        
        # Determine motive based on final verdict
        if final_verdict.lower() == "false":
            if primary_topic == "financial":
                primary_motive = "financial_manipulation"
                risk_level = "high"
                reasoning = "Content contains false financial information, likely intended to manipulate markets"
            else:
                primary_motive = "disinformation"
                risk_level = "high"
                reasoning = "Content contains false information, likely intended to mislead"
        
        elif final_verdict.lower() == "partially_true":
            # Check for temporal issues
            temporal_verdict = temporal_analysis.get("temporal_verdict", "unknown")
            intent_analysis = temporal_analysis.get("intent_analysis", "unknown")
            
            if "misleading" in intent_analysis.lower() or "old_news" in intent_analysis.lower():
                primary_motive = "outdated_information"
                risk_level = "moderate"
                reasoning = "Content is factually accurate but presented as current when it's outdated, potentially misleading"
            else:
                primary_motive = "partially_informing"
                risk_level = "moderate"
                reasoning = "Content has mixed accuracy, motive unclear"
        
        elif final_verdict.lower() == "true":
            # Check temporal appropriateness
            temporal_verdict = temporal_analysis.get("temporal_verdict", "appropriate")
            if temporal_verdict == "misleading_timing":
                primary_motive = "outdated_information"
                risk_level = "low"
                reasoning = "Content is true but timing may be misleading"
            else:
                primary_motive = "informing"
                risk_level = "low"
                reasoning = "Content appears to be genuinely informative with accurate information"
        
        elif final_verdict.lower() == "ironic":
            primary_motive = "satire"
            risk_level = "low"
            reasoning = "Content appears to be satirical or ironic in nature"
        
        else:  # inconclusive or unknown
            primary_motive = "unknown"
            risk_level = "moderate"
            reasoning = "Motive unclear due to inconclusive verdict"
        
        return {
            "primary_motive": primary_motive,
            "confidence_score": 0.7,
            "reasoning": reasoning,
            "risk_level": risk_level,
            "manipulation_indicators": self._identify_manipulation_indicators_from_verdict(final_verdict, primary_topic, temporal_analysis),
            "verdict_informed": True,
            "analysis_method": "rule_based_with_verdict"
        }
    
    def _identify_manipulation_indicators(self, fact_check_verdict: str, primary_topic: str) -> list:
        """Identify manipulation indicators based on fact-check results."""
        indicators = []
        
        if fact_check_verdict.lower() in ["false", "likely_false"]:
            indicators.append("False information detected")
            
            if primary_topic == "financial":
                indicators.append("Financial misinformation")
            elif primary_topic == "medical":
                indicators.append("Health misinformation")
            elif primary_topic == "political":
                indicators.append("Political misinformation")
        
        return indicators
    
    def _identify_manipulation_indicators_from_verdict(
        self, 
        final_verdict: str, 
        primary_topic: str, 
        temporal_analysis: Dict[str, Any]
    ) -> list:
        """Identify manipulation indicators based on final verdict and context."""
        indicators = []
        
        if final_verdict.lower() == "false":
            indicators.append("False information detected")
            if primary_topic == "financial":
                indicators.append("Financial misinformation")
            elif primary_topic == "medical":
                indicators.append("Health misinformation")
            elif primary_topic == "political":
                indicators.append("Political misinformation")
        
        elif final_verdict.lower() == "partially_true":
            intent_analysis = temporal_analysis.get("intent_analysis", "")
            if "misleading" in intent_analysis.lower():
                indicators.append("Temporal misleading detected")
            if "old_news" in intent_analysis.lower():
                indicators.append("Old information presented as current")
        
        return indicators
    
    def _enhance_analysis_with_context(
        self,
        analysis: Dict[str, Any],
        fact_check_verdict: str,
        primary_topic: str,
        temporal_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance analysis with additional context."""
        
        # Add fact-check context
        analysis["fact_check_verdict"] = fact_check_verdict
        analysis["primary_topic"] = primary_topic
        
        # Add temporal context
        if temporal_analysis:
            analysis["temporal_context"] = {
                "timing_assessment": temporal_analysis.get("timing_assessment", "unknown"),
                "temporal_verdict": temporal_analysis.get("temporal_verdict", "unknown")
            }
        
        # Ensure all required fields are present
        analysis.setdefault("primary_motive", "unknown")
        analysis.setdefault("confidence_score", 0.5)
        analysis.setdefault("reasoning", "Analysis completed")
        analysis.setdefault("risk_level", "moderate")
        analysis.setdefault("manipulation_indicators", [])
        
        return analysis
    
    def _enhance_analysis_with_verdict_context(
        self,
        analysis: Dict[str, Any],
        final_verdict: str,
        primary_topic: str,
        temporal_analysis: Dict[str, Any],
        post_date: str,
        mentioned_dates: list
    ) -> Dict[str, Any]:
        """Enhance analysis with verdict and temporal context."""
        
        # Add verdict context
        analysis["final_verdict"] = final_verdict
        analysis["primary_topic"] = primary_topic
        analysis["post_date"] = post_date
        analysis["mentioned_dates"] = mentioned_dates
        
        # Add temporal context
        if temporal_analysis:
            analysis["temporal_context"] = {
                "timing_assessment": temporal_analysis.get("timing_assessment", "unknown"),
                "temporal_verdict": temporal_analysis.get("temporal_verdict", "unknown"),
                "intent_analysis": temporal_analysis.get("intent_analysis", "unknown")
            }
        
        # Ensure all required fields are present
        analysis.setdefault("primary_motive", "unknown")
        analysis.setdefault("confidence_score", 0.5)
        analysis.setdefault("reasoning", "Analysis completed")
        analysis.setdefault("risk_level", "moderate")
        analysis.setdefault("manipulation_indicators", [])
        
        return analysis
    
    def _create_fallback_analysis(self, error_message: str) -> Dict[str, Any]:
        """Create a fallback analysis when normal analysis fails."""
        return {
            "primary_motive": "unknown",
            "confidence_score": 0.0,
            "reasoning": f"Analysis could not be completed: {error_message}",
            "risk_level": "moderate",
            "manipulation_indicators": [],
            "fact_check_informed": False,
            "analysis_method": "fallback"
        }