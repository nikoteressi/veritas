"""
Core agent implementation for Veritas fact-checking.
"""
import asyncio
import base64
import io
import json
import re
import logging
import hashlib
import time
from datetime import datetime
from typing import Dict, Any, AsyncGenerator, Optional, List

from langchain_core.output_parsers import JsonOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession

from agent.llm import llm_manager
from agent.tools import AVAILABLE_TOOLS
from agent.temporal_analysis import temporal_analyzer
from agent.vector_store import vector_store
from agent.prompts import (
    MULTIMODAL_ANALYSIS_PROMPT,
    FACT_CHECKING_PROMPT,
    VERDICT_GENERATION_PROMPT,
    DOMAIN_SPECIFIC_PROMPTS
)
from app.crud import UserCRUD, VerificationResultCRUD
from app.config import settings
from app.exceptions import LLMError, AgentError, ServiceUnavailableError
from app.schemas import ImageAnalysisResult

logger = logging.getLogger(__name__)


class VeritasAgent:
    """Main agent for fact-checking social media posts."""
    
    def __init__(self):
        self.llm = llm_manager.llm
        self.tools = AVAILABLE_TOOLS
        self.agent_executor = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the LangChain agent executor."""
        try:
            # Create a simple prompt for the agent
            from langchain_core.prompts import ChatPromptTemplate
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant with access to tools for fact-checking."),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            
            # Create the agent
            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=agent, 
                tools=self.tools, 
                verbose=True,
                max_iterations=10,
                early_stopping_method="generate"
            )
            
            logger.info("Agent executor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    async def verify_post(
        self,
        image_bytes: bytes,
        user_prompt: str,
        db: AsyncSession,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Main method to verify a social media post.
        
        Args:
            image_bytes: Raw image bytes
            user_prompt: User's question/prompt
            db: Database session
            progress_callback: Optional callback for progress updates
            
        Returns:
            Verification result dictionary
        """
        start_time = time.time()
        
        try:
            # Step 1: Multimodal Analysis
            if progress_callback:
                await progress_callback("Analyzing image content...", 10)
            
            analysis_result = await self._analyze_image(image_bytes, user_prompt)
            
            # The new _analyze_image returns a dictionary, so no extraction is needed.
            # We can use the result directly as extracted_info.
            extracted_info = analysis_result
            
            # Step 3: Get user reputation
            if progress_callback:
                await progress_callback("Checking user reputation...", 35)

            user_reputation = await self._get_user_reputation(
                db, extracted_info.get("username", "unknown")
            )

            # Step 3.5: Temporal analysis
            if progress_callback:
                await progress_callback("Analyzing temporal context...", 40)

            temporal_analysis = temporal_analyzer.analyze_temporal_context(extracted_info)
            extracted_info["temporal_analysis"] = temporal_analysis

            # Step 4: Fact-checking
            if progress_callback:
                await progress_callback("Fact-checking claims...", 50)
            
            fact_check_result = await self._fact_check_claims(
                extracted_info, user_prompt, progress_callback
            )
            
            # Step 5: Generate final verdict
            if progress_callback:
                await progress_callback("Generating final verdict...", 80)
            
            verdict_result = await self._generate_verdict(
                fact_check_result, user_prompt, extracted_info.get("temporal_analysis", {})
            )
            
            # Step 6: Update user reputation
            if progress_callback:
                await progress_callback("Updating user reputation...", 90)
            
            updated_reputation = await self._update_user_reputation(
                db, extracted_info.get("username", "unknown"), verdict_result["verdict"]
            )
            
            # Step 7: Save verification result
            if progress_callback:
                await progress_callback("Saving results...", 95)
            
            verification_record = await self._save_verification_result(
                db, image_bytes, user_prompt, extracted_info, verdict_result
            )

            # Step 7.5: Store in vector database and check for similar verifications
            await self._store_in_vector_db(extracted_info, verdict_result, fact_check_result)

            # Step 8: Compile final result
            processing_time = int(time.time() - start_time)
            
            final_result = {
                "status": "success",
                "message": "Verification completed successfully",
                "verification_id": str(verification_record.id),  # Convert to string for consistency
                "user_nickname": extracted_info.get("username"),
                "extracted_text": extracted_info.get("extracted_text"),
                "primary_topic": extracted_info.get("primary_topic"),
                "identified_claims": extracted_info.get("claims", []),
                "verdict": verdict_result["verdict"],
                "justification": verdict_result["justification"],
                "confidence_score": verdict_result["confidence_score"],
                "processing_time_seconds": processing_time,
                "temporal_analysis": extracted_info.get("temporal_analysis", {}),
                "examined_sources": fact_check_result.get("examined_sources", []),
                "search_queries_used": fact_check_result.get("search_queries_used", []),
                "fact_check_summary": {
                    "total_sources_found": fact_check_result.get("total_sources_found", 0),
                    "credible_sources": fact_check_result.get("credible_sources", 0),
                    "supporting_evidence": fact_check_result.get("supporting_evidence", 0),
                    "contradicting_evidence": fact_check_result.get("contradicting_evidence", 0)
                },
                "user_reputation": {
                    "nickname": updated_reputation.nickname,
                    "true_count": updated_reputation.true_count,
                    "partially_true_count": updated_reputation.partially_true_count,
                    "false_count": updated_reputation.false_count,
                    "ironic_count": updated_reputation.ironic_count,
                    "total_posts_checked": updated_reputation.total_posts_checked,
                    "warning_issued": updated_reputation.warning_issued,
                    "notification_issued": updated_reputation.notification_issued
                },
                "warnings": self._generate_warnings(updated_reputation)
            }
            
            if progress_callback:
                await progress_callback("Verification complete!", 100)
            
            logger.info(f"Verification completed in {processing_time}s: {verdict_result['verdict']}")
            return final_result
            
        except Exception as e:
            logger.error(f"Verification failed: {e}", exc_info=True)
            raise  # Re-raise the exception to be handled by global error handlers
    
    async def _analyze_image(self, image_bytes: bytes, user_prompt: str) -> Dict[str, Any]:
        """Analyze the image using multimodal LLM and return a structured dictionary."""
        
        output_parser = JsonOutputParser(pydantic_object=ImageAnalysisResult)

        try:
            current_date = datetime.now().strftime("%Y-%m-%d")

            prompt = MULTIMODAL_ANALYSIS_PROMPT.partial(
                format_instructions=output_parser.get_format_instructions()
            )
            
            # Format the final prompt with user input
            final_prompt = await prompt.aformat(user_prompt=user_prompt)
            
            llm_output = await llm_manager.invoke_multimodal(final_prompt, image_bytes)
            
            parsed_output = await output_parser.aparse(llm_output)

            # Inject additional context that wasn't part of the initial model analysis
            parsed_output['contextual_information'] = {
                "current_date": current_date,
                "user_prompt": user_prompt
            }

            logger.debug(f"Image analysis result: {json.dumps(parsed_output, indent=2)}")
            logger.info("Image analysis completed and JSON extracted")
            return parsed_output
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}", exc_info=True)
            raise LLMError(
                f"Image analysis failed during parsing: {e}", 
                error_code="IMAGE_ANALYSIS_FAILED"
            )
    
    async def _get_user_reputation(self, db: AsyncSession, nickname: str) -> Any:
        """Get user reputation from database."""
        return await UserCRUD.get_or_create_user(db, nickname)
    
    async def _fact_check_claims(
        self, 
        extracted_info: Dict[str, Any], 
        user_prompt: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Perform fact-checking using available tools."""
        try:
            # Create fact-checking prompt
            domain = extracted_info.get("primary_topic", "general")
            
            # Add domain-specific instructions if available
            domain_prompt = DOMAIN_SPECIFIC_PROMPTS.get(domain, "")
            
            fact_check_input = {
                "post_analysis": extracted_info["extracted_text"],
                "claims": extracted_info.get("claims", []),
                "user_prompt": user_prompt,
                "domain": domain,
                "temporal_context": json.dumps(extracted_info.get("temporal_analysis", {}))
            }
            
            # Use the agent executor for fact-checking
            if self.agent_executor:
                result = await self.agent_executor.ainvoke({
                    "input": f"Fact-check this social media post: {json.dumps(fact_check_input)}"
                })
                
                return {
                    "domain": domain,
                    "research_results": result.get("output", ""),
                    "tools_used": ["agent_executor"]
                }
            else:
                # Fallback to direct LLM call
                prompt_text = f"Fact-check this content: {extracted_info['extracted_text']}"
                result = await llm_manager.invoke_text_only(prompt_text)
                
                return {
                    "domain": domain,
                    "research_results": result,
                    "tools_used": ["direct_llm"]
                }
                
        except Exception as e:
            logger.error(f"Fact-checking failed: {e}")
            return {
                "domain": "general",
                "research_results": f"Fact-checking failed: {e}",
                "tools_used": []
            }
    
    async def _generate_verdict(
        self, 
        fact_check_result: Dict[str, Any],
        user_prompt: str,
        temporal_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate the final verdict based on fact-checking results."""
        try:
            # Prepare the research summary
            research_summary = self._summarize_fact_check(fact_check_result)
            
            # Format the prompt
            prompt = await VERDICT_GENERATION_PROMPT.aformat(
                research_results=research_summary,
                user_prompt=user_prompt,
                temporal_analysis=json.dumps(temporal_analysis, indent=2)
            )
            
            # Invoke the LLM
            llm_response = await llm_manager.invoke_text_only(prompt)
            
            # Parse the response to get structured verdict
            parsed_verdict = self._parse_verdict_response(llm_response)
            
            logger.info(f"Generated verdict: {parsed_verdict.get('verdict')}")
            return parsed_verdict
            
        except Exception as e:
            logger.error(f"Failed to generate verdict: {e}", exc_info=True)
            return {
                "verdict": "partially_true",
                "justification": "Could not generate a definitive verdict due to an internal error.",
                "confidence_score": 0
            }
    
    def _summarize_fact_check(self, fact_check_result: Dict[str, Any]) -> str:
        """Create a concise summary of the fact-checking results."""
        summary_parts = []
        
        assessment = fact_check_result.get('preliminary_assessment', 'N/A')
        summary_parts.append(f"- Preliminary Assessment: {assessment}")

        confidence = fact_check_result.get('confidence_score', 'N/A')
        summary_parts.append(f"- Confidence Score: {confidence}")

        supporting = fact_check_result.get('supporting_evidence', 0)
        contradicting = fact_check_result.get('contradicting_evidence', 0)
        summary_parts.append(f"- Evidence: Found {supporting} supporting and {contradicting} contradicting pieces of evidence.")

        credible_sources = fact_check_result.get('credible_sources', 0)
        summary_parts.append(f"- Sources: Utilized {credible_sources} credible sources out of {fact_check_result.get('total_sources_found', 0)} total.")

        if 'confidence_factors' in fact_check_result:
            factors = ", ".join(fact_check_result['confidence_factors'])
            summary_parts.append(f"- Confidence Factors: {factors}")

        return "\n".join(summary_parts)

    def _parse_verdict_response(self, response: str) -> Dict[str, Any]:
        """Parse the structured verdict from the LLM's text response."""
        verdict_match = re.search(r"^\s*\*\*Verdict(?: Classification)?:\*\*\s*(true|partially_true|false|ironic)", response, re.IGNORECASE | re.MULTILINE)
        confidence_match = re.search(r"\*\*Confidence Score:\*\*\s*(\d+)", response, re.IGNORECASE | re.MULTILINE)
        justification_match = re.search(r"\*\*Justification:\*\*\s*(.*?)(\*\*Key Sources:\*\*|\*\*Caveats and Limitations:\*\*|$)", response, re.IGNORECASE | re.DOTALL)
        sources_match = re.search(r"\*\*Key Sources:\*\*\s*(.*?)(\*\*Caveats and Limitations:\*\*|$)", response, re.IGNORECASE | re.DOTALL)

        verdict = verdict_match.group(1).lower() if verdict_match else "partially_true"
        confidence = int(confidence_match.group(1)) if confidence_match else 50
        justification = justification_match.group(1).strip() if justification_match else "No justification provided."
        sources_str = sources_match.group(1).strip() if sources_match else "No sources cited."

        return {
            "verdict": verdict,
            "confidence_score": confidence,
            "justification": justification,
            "sources": sources_str
        }
    
    async def _update_user_reputation(
        self, 
        db: AsyncSession, 
        nickname: str, 
        verdict: str
    ) -> Any:
        """Update user reputation in database."""
        return await UserCRUD.update_user_reputation(db, nickname, verdict)
    
    async def _save_verification_result(
        self,
        db: AsyncSession,
        image_bytes: bytes,
        user_prompt: str,
        extracted_info: Dict[str, Any],
        verdict_result: Dict[str, Any]
    ) -> Any:
        """Save verification result to database."""
        try:
            # Generate image hash
            image_hash = hashlib.sha256(image_bytes).hexdigest()
            
            verification_record = await VerificationResultCRUD.create_verification_result(
                db=db,
                user_nickname=extracted_info.get("username", "unknown"),
                image_hash=image_hash,
                extracted_text=extracted_info.get("extracted_text", ""),
                user_prompt=user_prompt,
                primary_topic=extracted_info.get("primary_topic", "general"),
                identified_claims=json.dumps(extracted_info.get("claims", [])),
                verdict=verdict_result["verdict"],
                justification=verdict_result["justification"],
                confidence_score=verdict_result["confidence_score"],
                processing_time_seconds=0,  # Will be updated
                model_used=settings.ollama_model,
                tools_used=json.dumps(verdict_result.get("sources", []))
            )
            return verification_record
        except Exception as e:
            logger.error(f"Failed to save verification result: {e}", exc_info=True)
            # Depending on requirements, you might want to raise an exception here
            return None # Or handle it gracefully

    def _generate_warnings(self, user_reputation: Any) -> List[str]:
        """Generate warnings based on user's reputation."""
        warnings = []
        
        if user_reputation.total_posts_checked < 5:
            warnings.append("User has a limited history, so their reputation may not be fully representative.")
        
        if user_reputation.false_count > 3:
            warnings.append("User has a history of posting false information.")
        
        if user_reputation.warning_issued:
            warnings.append("A warning has been previously issued to this user for spreading misinformation.")
            
        return warnings

    async def _store_in_vector_db(
        self,
        extracted_info: Dict[str, Any],
        verdict_result: Dict[str, Any],
        fact_check_result: Dict[str, Any]
    ):
        """Store verification result in vector database asynchronously."""
        try:
            # Run vector storage in background to avoid blocking
            import asyncio

            def store_in_background():
                try:
                    # Prepare data for vector storage
                    vector_data = {
                        "nickname": extracted_info.get("username"),
                        "extracted_text": extracted_info.get("extracted_text"),
                        "claims": extracted_info.get("claims", []),
                        "temporal_analysis": extracted_info.get("temporal_analysis", {}),
                        "verdict": verdict_result.get("verdict"),
                        "confidence_score": verdict_result.get("confidence_score"),
                        "justification": verdict_result.get("justification"),
                        "fact_check_results": fact_check_result
                    }

                    # Store in vector database (non-blocking)
                    verification_id = vector_store.store_verification_result(vector_data)
                    if verification_id:
                        logger.info(f"Stored verification in vector database: {verification_id}")

                    # Check for similar verifications (non-blocking)
                    if extracted_info.get("claims"):
                        for claim in extracted_info["claims"]:
                            similar_claims = vector_store.find_similar_claims(claim, limit=3)
                            if similar_claims:
                                logger.info(f"Found {len(similar_claims)} similar claims for: {claim[:50]}...")

                except Exception as e:
                    logger.warning(f"Background vector storage failed: {e}")

            # Run in background thread to avoid blocking main process
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, store_in_background)

            logger.info("Vector storage initiated in background")

        except Exception as e:
            logger.warning(f"Failed to initiate vector storage: {e}")
            # Don't raise error - vector storage is not critical for main functionality


# Global agent instance
veritas_agent = VeritasAgent()
