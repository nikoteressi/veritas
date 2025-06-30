import json
from typing import Dict, Any, Optional
from log import logger
from agent.errors import AgentError, LLMError
from user.crud import UserCRUD
from llm.manager import llm_manager

class Core:
    def __init__(self, agent_executor):
        self.agent_executor = agent_executor

    async def _fact_check_claims(
        self, 
        extracted_info: Dict[str, Any], 
        user_prompt: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Perform fact-checking using available tools."""
        claims = extracted_info.get("claims", [])
        primary_topic = extracted_info.get("primary_topic", "general")
        
        if not claims:
            logger.warning("No claims extracted, skipping fact-checking.")
            return {"verdict": "no_claims", "justification": "No specific claims were identified in the content."}

        # For simplicity, we'll focus on the first claim for now
        # In a real scenario, you might want to iterate or combine them
        claim_to_check = claims[0]
        context = json.dumps({
            "user_prompt": user_prompt,
            "all_claims": claims,
            "temporal_analysis": extracted_info.get("temporal_analysis", {})
        })

        try:
            # Invoke the agent with the fact-checking tool
            tool_input = {
                "claim": claim_to_check,
                "context": context,
                "domain": primary_topic
            }
            
            # The agent executor's output is a dictionary, and the relevant result is in the 'output' key
            response = await self.agent_executor.ainvoke({"input": f"Fact-check this claim: {claim_to_check}", "tool_input": tool_input})
            
            # The output from the tool is a JSON string, so we need to parse it
            fact_check_result_str = response.get("output", "{}")
            fact_check_result = json.loads(fact_check_result_str)
            
            logger.info(f"Fact-checking for claim '{claim_to_check}' completed.")
            return fact_check_result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse fact-checking result JSON: {e}")
            raise AgentError("Fact-checking returned invalid JSON.", "JSON_DECODE_ERROR")
        except Exception as e:
            logger.error(f"Fact-checking failed: {e}", exc_info=True)
            raise AgentError(f"An error occurred during fact-checking: {e}", "FACT_CHECK_ERROR")

    async def _generate_verdict(
        self, 
        fact_check_result: Dict[str, Any],
        user_prompt: str,
        temporal_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a final verdict based on all gathered evidence."""
        
        summary = self._summarize_fact_check(fact_check_result)
        
        prompt = VERDICT_GENERATION_PROMPT.partial()
        
        final_prompt_str = await prompt.aformat(
            user_question=user_prompt,
            temporal_analysis=json.dumps(temporal_analysis, indent=2),
            fact_check_summary=summary
        )
        
        logger.debug(f"Invoking text-only LLM with prompt:\n---PROMPT START---\n{final_prompt_str}\n---PROMPT END---")
        
        try:
            response = await llm_manager.invoke_text_only(final_prompt_str)
            logger.debug(f"LLM text-only response:\n---RESPONSE START---\n{response}\n---RESPONSE END---")
            
            parsed_verdict = self._parse_verdict_response(response)
            
            logger.info(f"Generated verdict: {parsed_verdict.get('verdict')}")
            return parsed_verdict
            
        except Exception as e:
            logger.error(f"Failed to generate final verdict: {e}", exc_info=True)
            raise LLMError("Failed to generate final verdict.", "VERDICT_GENERATION_FAILED")

    def _summarize_fact_check(self, fact_check_result: Dict[str, Any]) -> str:
        """Create a concise summary of the fact-checking results for the final prompt."""
        if not fact_check_result:
            return "No fact-checking was performed."

        preliminary_assessment = fact_check_result.get("preliminary_assessment", "N/A")
        confidence_score = fact_check_result.get("confidence_score", "N/A")
        supporting = fact_check_result.get("supporting_evidence", 0)
        contradicting = fact_check_result.get("contradicting_evidence", 0)
        credible_sources = fact_check_result.get("credible_sources", 0)
        total_sources = fact_check_result.get("total_sources_found", 0)

        summary = (
            f"- Preliminary Assessment: {preliminary_assessment}\n"
            f"- Confidence Score: {confidence_score}\n"
            f"- Evidence: Found {supporting} supporting and {contradicting} contradicting pieces of evidence.\n"
            f"- Sources: Utilized {credible_sources} credible sources out of {total_sources} total."
        )
        return summary

    def _parse_verdict_response(self, response: str) -> Dict[str, Any]:
        """Parse the structured response from the verdict generation LLM."""
        # Implementation of _parse_verdict_response method
        pass

    async def get_or_create_user(self, db, nickname):
        return await UserCRUD.get_or_create_user(db, nickname) 