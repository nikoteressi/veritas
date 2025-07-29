"""
from __future__ import annotations

Refactored motives analysis module for analyzing potential motives behind social media posts.
Now uses fact-check verdicts and temporal analysis as primary inputs for informed motive determination.
"""

import json
import logging
import re
from typing import Any

from agent.analyzers.base_analyzer import BaseAnalyzer
from agent.models.motives_analysis import MotivesAnalysisResult
from agent.models.temporal_analysis import TemporalAnalysisResult
from agent.models.verification_context import VerificationContext
from app.exceptions import MotivesAnalysisError
from langchain_core.output_parsers import PydanticOutputParser

logger = logging.getLogger(__name__)


class MotivesAnalyzer(BaseAnalyzer):
    """Analyzes potential motives behind social media posts using an LLM."""

    def __init__(self, llm_manager, prompt_manager):
        super().__init__("motives")
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager

    async def analyze(self, context: VerificationContext) -> MotivesAnalysisResult:
        """Analyzes motives using the reasoning LLM and context data, including summarization results."""
        temporal_analysis = context.get_temporal_analysis()
        summarization_result = context.get_summarization_result()

        # Check for required data - now including summarization
        if not all([context.screenshot_data, temporal_analysis]):
            raise MotivesAnalysisError("Missing required data in context for motives analysis.")

        output_parser = PydanticOutputParser(pydantic_object=MotivesAnalysisResult)

        try:
            # Prepare enhanced context with summarization
            summary_text = summarization_result.summary if summarization_result else "No summary available"
            key_points = summarization_result.key_points if summarization_result else []
            confidence_score = summarization_result.confidence_score if summarization_result else 0.5

            # Format key points for better readability
            key_points_text = (
                "\n".join([f"• {point}" for point in key_points]) if key_points else "No key points identified"
            )

            prompt_template = self.prompt_manager.get_prompt_template("motives_analysis_enhanced")
            prompt = await prompt_template.aformat(
                # Core context
                temporal_analysis=temporal_analysis,
                screenshot_data=context.screenshot_data,
                # Enhanced with summarization - matching template variable names
                summary_text=summary_text,
                key_points=key_points_text,
                confidence_score=confidence_score,
                # Fact-check context if available
                fact_check_result=context.fact_check_result,
                # Format instructions
                format_instructions=output_parser.get_format_instructions(),
            )

            logger.info("Enhanced motives analysis prompt prepared with summarization context")

            response = await self.llm_manager.invoke_text_only(prompt)
            parsed_response = output_parser.parse(response)

            if not isinstance(parsed_response, MotivesAnalysisResult):
                raise MotivesAnalysisError("Failed to parse motive analysis from LLM response.")

            logger.info("Successfully performed enhanced LLM-based motives analysis with summarization context.")
            return parsed_response

        except MotivesAnalysisError:
            raise
        except (
            ValueError,
            RuntimeError,
            AttributeError,
            TypeError,
            json.JSONDecodeError,
        ) as e:
            logger.error(f"Error during enhanced motives analysis: {e}", exc_info=True)
            raise MotivesAnalysisError(f"Enhanced motives analysis failed: {e}") from e

    def _extract_fact_check_verdict(self, fact_check_result) -> str:
        """Extract the fact-check verdict from the result."""
        if hasattr(fact_check_result, "claim_results") and fact_check_result.claim_results:
            # Look for overall assessment in claim results
            for claim_result in fact_check_result.claim_results:
                if "assessment" in claim_result:
                    logger.info(f"Fact-check verdict: {claim_result['summary']}")
                    return claim_result["summary"]

        # Look for verdict in summary
        if hasattr(fact_check_result, "summary"):
            summary = fact_check_result.summary
            if hasattr(summary, "model_dump"):
                summary_dict = summary.model_dump()
                for key, value in summary_dict.items():
                    if "assessment" in key.lower() or "verdict" in key.lower():
                        logger.info(f"Fact-check verdict2: {value}")
                        return str(value)

        return "unknown"

    def _extract_fact_check_confidence(self, fact_check_result) -> float:
        """Extract confidence score from fact-check results."""
        if hasattr(fact_check_result, "claim_results") and fact_check_result.claim_results:
            confidences = []
            for claim_result in fact_check_result.claim_results:
                if "confidence" in claim_result:
                    confidences.append(claim_result["confidence"])

            if confidences:
                # Normalize to 0-1
                return sum(confidences) / len(confidences) / 100.0

        return 0.5  # Default moderate confidence

    async def _analyze_motives_with_llm(
        self,
        content: str,
        fact_check_verdict: str,
        fact_check_confidence: float,
        temporal_analysis: TemporalAnalysisResult | None,
        primary_topic: str,
    ) -> dict[str, Any]:
        """Use LLM to analyze motives based on fact-check results."""

        # Format temporal analysis for readability
        temporal_summary = self._format_temporal_analysis(temporal_analysis)

        # Create prompt with fact-check context
        prompt_template = self.prompt_manager.get_prompt_template("motives_analysis")
        prompt = await prompt_template.aformat(
            fact_check_verdict=fact_check_verdict,
            fact_check_confidence=fact_check_confidence,
            temporal_analysis=temporal_summary,
            primary_topic=primary_topic,
            content=content,
        )
        logger.info(f"Motives analysis prompt: {prompt}")

        # Get LLM response
        response = await self.llm_manager.invoke_text_only(prompt)

        # Parse JSON response
        try:
            # Clean response to extract JSON
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
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
                    "analysis_method": "fact_check_informed_llm",
                }
        except Exception as parse_error:
            logger.warning(f"Failed to parse LLM motives analysis: {parse_error}")
            raise MotivesAnalysisError(f"Failed to parse LLM motives analysis: {parse_error}") from parse_error

        # Fallback to rule-based analysis if LLM parsing fails
        return self._rule_based_analysis(fact_check_verdict, primary_topic, temporal_analysis)

    async def _analyze_motives_with_verdict(
        self,
        content: str,
        final_verdict: str,
        verdict_confidence: float,
        verdict_reasoning: str,
        temporal_analysis: TemporalAnalysisResult | None,
        primary_topic: str,
        post_date: str,
        mentioned_dates: list,
    ) -> dict[str, Any]:
        """Use LLM to analyze motives based on final verdict."""

        # Format temporal analysis for readability
        temporal_summary = self._format_temporal_analysis(temporal_analysis)

        # Format dates information
        dates_info = (
            f"Post Date: {post_date}, Mentioned Dates: {', '.join(mentioned_dates) if mentioned_dates else 'None'}"
        )

        # Create prompt with verdict context
        prompt_template = self.prompt_manager.get_prompt_template("motives_analysis")
        prompt = await prompt_template.aformat(
            fact_check_verdict=f"{final_verdict} (Confidence: {verdict_confidence:.2f})",
            fact_check_confidence=verdict_confidence,
            temporal_analysis=f"{temporal_summary} | {dates_info}",
            primary_topic=primary_topic,
            content=content,
        )

        # Add verdict reasoning as additional context
        full_prompt = f"{prompt}\n\n**Verdict Reasoning**: {verdict_reasoning}"

        logger.info(f"Motives analysis prompt: {full_prompt}")

        # Get LLM response
        response = await self.llm_manager.invoke_text_only(full_prompt)

        # Parse JSON response
        try:
            # Clean response to extract JSON
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
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
                    "analysis_method": "verdict_informed_llm",
                }
        except Exception as parse_error:
            logger.warning(f"Failed to parse LLM motives analysis: {parse_error}")
            raise MotivesAnalysisError(f"Failed to parse LLM motives analysis: {parse_error}") from parse_error

        # Fallback to rule-based analysis if LLM parsing fails
        return self._rule_based_analysis_with_verdict(
            final_verdict, primary_topic, temporal_analysis, post_date, mentioned_dates
        )

    def _format_temporal_analysis(self, temporal_analysis: TemporalAnalysisResult | None) -> str:
        """Format temporal analysis for prompt readability."""
        if not temporal_analysis:
            return "No temporal analysis available"

        # Extract key information from typed object
        post_date = temporal_analysis.post_date or "unknown"
        temporal_context = temporal_analysis.temporal_context or "unknown"
        date_relevance = temporal_analysis.date_relevance or "unknown"

        return f"Post Date: {post_date}, Context: {temporal_context}, Relevance: {date_relevance}"

    def _rule_based_analysis(
        self,
        fact_check_verdict: str,
        primary_topic: str,
        temporal_analysis: TemporalAnalysisResult | None,
    ) -> dict[str, Any]:
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
            date_relevance = temporal_analysis.date_relevance if temporal_analysis else "appropriate"
            if "misleading" in date_relevance.lower() or "outdated" in date_relevance.lower():
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
            "analysis_method": "rule_based_with_fact_check",
        }

    def _rule_based_analysis_with_verdict(
        self,
        final_verdict: str,
        primary_topic: str,
        temporal_analysis: TemporalAnalysisResult | None,
        post_date: str,
        mentioned_dates: list,
    ) -> dict[str, Any]:
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
            temporal_context = temporal_analysis.temporal_context if temporal_analysis else "unknown"

            if "misleading" in temporal_context.lower() or "outdated" in temporal_context.lower():
                primary_motive = "outdated_information"
                risk_level = "moderate"
                reasoning = (
                    "Content is factually accurate but presented as current when it's outdated, potentially misleading"
                )
            else:
                primary_motive = "partially_informing"
                risk_level = "moderate"
                reasoning = "Content has mixed accuracy, motive unclear"

        elif final_verdict.lower() == "true":
            # Check temporal appropriateness
            date_relevance = temporal_analysis.date_relevance if temporal_analysis else "appropriate"
            if "misleading" in date_relevance.lower() or "outdated" in date_relevance.lower():
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
            "manipulation_indicators": self._identify_manipulation_indicators_from_verdict(
                final_verdict, primary_topic, temporal_analysis
            ),
            "verdict_informed": True,
            "analysis_method": "rule_based_with_verdict",
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
        temporal_analysis: TemporalAnalysisResult | None,
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
            temporal_context = temporal_analysis.temporal_context if temporal_analysis else ""
            if "misleading" in temporal_context.lower():
                indicators.append("Temporal misleading detected")
            if "outdated" in temporal_context.lower():
                indicators.append("Old information presented as current")

        return indicators

    def _enhance_analysis_with_context(
        self,
        analysis: dict[str, Any],
        fact_check_verdict: str,
        primary_topic: str,
        temporal_analysis: TemporalAnalysisResult | None,
    ) -> dict[str, Any]:
        """Enhance analysis with additional context."""

        # Add fact-check context
        analysis["fact_check_verdict"] = fact_check_verdict
        analysis["primary_topic"] = primary_topic

        # Add temporal context
        if temporal_analysis:
            analysis["temporal_context"] = {
                "post_date": temporal_analysis.post_date or "unknown",
                "temporal_context": temporal_analysis.temporal_context or "unknown",
                "date_relevance": temporal_analysis.date_relevance or "unknown",
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
        analysis: dict[str, Any],
        final_verdict: str,
        primary_topic: str,
        temporal_analysis: TemporalAnalysisResult | None,
        post_date: str,
        mentioned_dates: list,
    ) -> dict[str, Any]:
        """Enhance analysis with verdict and temporal context."""

        # Add verdict context
        analysis["final_verdict"] = final_verdict
        analysis["primary_topic"] = primary_topic
        analysis["post_date"] = post_date
        analysis["mentioned_dates"] = mentioned_dates

        # Add temporal context
        if temporal_analysis:
            analysis["temporal_context"] = {
                "post_date": temporal_analysis.post_date or "unknown",
                "temporal_context": temporal_analysis.temporal_context or "unknown",
                "date_relevance": temporal_analysis.date_relevance or "unknown",
            }

        # Ensure all required fields are present
        analysis.setdefault("primary_motive", "unknown")
        analysis.setdefault("confidence_score", 0.5)
        analysis.setdefault("reasoning", "Analysis completed")
        analysis.setdefault("risk_level", "moderate")
        analysis.setdefault("manipulation_indicators", [])

        return analysis

    def _create_fallback_analysis(self, error_message: str) -> dict[str, Any]:
        """Create a fallback analysis when normal analysis fails."""
        return {
            "primary_motive": "unknown",
            "confidence_score": 0.0,
            "reasoning": f"Analysis could not be completed: {error_message}",
            "risk_level": "moderate",
            "manipulation_indicators": [],
            "fact_check_informed": False,
            "analysis_method": "fallback",
        }

    async def _fallback_analysis(self, context: VerificationContext) -> MotivesAnalysisResult:
        """Fallback analysis method when enhanced analysis fails."""
        try:
            logger.warning("Using fallback analysis for motives")

            # Basic rule-based analysis using available context
            temporal_analysis = context.get_temporal_analysis()

            # Determine basic motive based on available information
            primary_motive = "unknown"
            risk_level = "moderate"
            reasoning = "Basic analysis performed due to enhanced analysis failure"
            confidence_score = 0.3
            manipulation_indicators = []

            # Try to use fact-check results if available
            if context.fact_check_result:
                fact_check_verdict = self._extract_fact_check_verdict(context.fact_check_result)
                if fact_check_verdict.lower() == "false":
                    primary_motive = "disinformation"
                    risk_level = "high"
                    reasoning = "Content contains false information based on fact-check"
                    manipulation_indicators = ["False information detected"]
                elif fact_check_verdict.lower() == "true":
                    primary_motive = "informing"
                    risk_level = "low"
                    reasoning = "Content appears to be informative based on fact-check"

            # Determine credibility assessment based on risk level
            if risk_level == "high":
                credibility_assessment = "low"
            elif risk_level == "low":
                credibility_assessment = "high"
            else:
                credibility_assessment = "moderate"

            # Create result object
            return MotivesAnalysisResult(
                primary_motive=primary_motive,
                confidence_score=confidence_score,
                credibility_assessment=credibility_assessment,
                risk_level=risk_level,
                manipulation_indicators=manipulation_indicators,
                analysis_summary=reasoning,
            )

        except Exception as e:
            logger.error("Even fallback analysis failed: %s", e)
            raise MotivesAnalysisError(f"Fallback motives analysis failed: {e}") from e
