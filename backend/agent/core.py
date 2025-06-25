"""
Core agent implementation for Veritas fact-checking.
"""
import json
import logging
import hashlib
import time
from typing import Dict, Any, AsyncGenerator, Optional, List

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession

from agent.llm import llm_manager
from agent.tools import AVAILABLE_TOOLS
from agent.prompts import (
    MULTIMODAL_ANALYSIS_PROMPT,
    FACT_CHECKING_PROMPT,
    VERDICT_GENERATION_PROMPT,
    DOMAIN_SPECIFIC_PROMPTS
)
from app.crud import UserCRUD, VerificationResultCRUD
from app.config import settings
from app.exceptions import LLMError, AgentError, ServiceUnavailableError

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
            
            # Step 2: Extract key information
            if progress_callback:
                await progress_callback("Extracting claims and user information...", 25)
            
            extracted_info = self._extract_key_information(analysis_result)
            
            # Step 3: Get user reputation
            if progress_callback:
                await progress_callback("Checking user reputation...", 35)
            
            user_reputation = await self._get_user_reputation(
                db, extracted_info.get("nickname", "unknown")
            )
            
            # Step 4: Fact-checking
            if progress_callback:
                await progress_callback("Fact-checking claims...", 50)
            
            fact_check_result = await self._fact_check_claims(
                extracted_info, user_prompt, progress_callback
            )
            
            # Step 5: Generate final verdict
            if progress_callback:
                await progress_callback("Generating final verdict...", 80)
            
            verdict_result = await self._generate_verdict(fact_check_result)
            
            # Step 6: Update user reputation
            if progress_callback:
                await progress_callback("Updating user reputation...", 90)
            
            updated_reputation = await self._update_user_reputation(
                db, extracted_info.get("nickname", "unknown"), verdict_result["verdict"]
            )
            
            # Step 7: Save verification result
            if progress_callback:
                await progress_callback("Saving results...", 95)
            
            verification_record = await self._save_verification_result(
                db, image_bytes, user_prompt, extracted_info, verdict_result
            )
            
            # Step 8: Compile final result
            processing_time = int(time.time() - start_time)
            
            final_result = {
                "status": "success",
                "message": "Verification completed successfully",
                "verification_id": str(verification_record.id),  # Convert to string for consistency
                "user_nickname": extracted_info.get("nickname"),
                "extracted_text": extracted_info.get("extracted_text"),
                "primary_topic": extracted_info.get("primary_topic"),
                "identified_claims": extracted_info.get("claims", []),
                "verdict": verdict_result["verdict"],
                "justification": verdict_result["justification"],
                "confidence_score": verdict_result["confidence_score"],
                "processing_time_seconds": processing_time,
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

            # Attempt graceful degradation
            processing_time = int(time.time() - start_time)

            # Try to provide partial results if possible
            try:
                if progress_callback:
                    await progress_callback("Error occurred, attempting fallback analysis...", 50)

                # Basic fallback analysis
                fallback_result = await self._fallback_analysis(image_bytes, user_prompt, db)
                fallback_result.update({
                    "status": "partial_success",
                    "message": f"Primary analysis failed, fallback analysis completed. Error: {str(e)}",
                    "processing_time_seconds": processing_time,
                    "warnings": ["Analysis completed with limited functionality due to service issues"]
                })

                if progress_callback:
                    await progress_callback("Fallback analysis completed", 100)

                return fallback_result

            except Exception as fallback_error:
                logger.error(f"Fallback analysis also failed: {fallback_error}")

                return {
                    "status": "error",
                    "message": f"Verification failed: {str(e)}",
                    "processing_time_seconds": processing_time,
                    "error_details": {
                        "primary_error": str(e),
                        "fallback_error": str(fallback_error)
                    }
                }
    
    async def _analyze_image(self, image_bytes: bytes, user_prompt: str) -> str:
        """Analyze the image using multimodal LLM."""
        try:
            # Create the analysis prompt
            prompt_text = MULTIMODAL_ANALYSIS_PROMPT.format_messages(
                user_prompt=user_prompt
            )[1].content
            
            # Invoke multimodal LLM
            result = await llm_manager.invoke_multimodal(prompt_text, image_bytes)
            
            logger.info("Image analysis completed")
            return result
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise
    
    def _extract_key_information(self, analysis_result: str) -> Dict[str, Any]:
        """Extract structured information from analysis result."""
        # This is a simplified extraction
        # In a real implementation, you'd use more sophisticated parsing
        
        extracted = {
            "extracted_text": "",
            "nickname": "unknown",
            "primary_topic": "general",
            "claims": [],
            "irony_assessment": "not_ironic"
        }
        
        # Simple keyword-based extraction (placeholder)
        lines = analysis_result.split('\n')
        for line in lines:
            if "nickname" in line.lower() or "username" in line.lower():
                # Extract nickname (simplified)
                parts = line.split(':')
                if len(parts) > 1:
                    extracted["nickname"] = parts[1].strip()
            elif "topic" in line.lower() or "category" in line.lower():
                # Extract topic (simplified)
                for topic in ["medical", "financial", "political", "scientific"]:
                    if topic in line.lower():
                        extracted["primary_topic"] = topic
                        break
        
        extracted["extracted_text"] = analysis_result
        
        return extracted
    
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
                "user_prompt": user_prompt,
                "domain": domain
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
    
    async def _generate_verdict(self, fact_check_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final verdict based on fact-checking results."""
        try:
            prompt_text = VERDICT_GENERATION_PROMPT.format_messages(
                research_results=fact_check_result["research_results"]
            )[1].content
            
            result = await llm_manager.invoke_text_only(prompt_text)
            
            # Parse verdict (simplified)
            verdict = "partially_true"  # Default
            confidence = 50  # Default
            
            # Simple parsing (in practice, you'd use more sophisticated extraction)
            if "true" in result.lower() and "false" not in result.lower():
                verdict = "true"
                confidence = 80
            elif "false" in result.lower():
                verdict = "false"
                confidence = 75
            elif "ironic" in result.lower() or "satirical" in result.lower():
                verdict = "ironic"
                confidence = 90
            
            return {
                "verdict": verdict,
                "justification": result,
                "confidence_score": confidence,
                "sources": fact_check_result.get("tools_used", [])
            }
            
        except Exception as e:
            logger.error(f"Verdict generation failed: {e}")
            return {
                "verdict": "partially_true",
                "justification": f"Verdict generation failed: {e}",
                "confidence_score": 0,
                "sources": []
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
        # Generate image hash
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        
        return await VerificationResultCRUD.create_verification_result(
            db=db,
            user_nickname=extracted_info.get("nickname", "unknown"),
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
    
    def _generate_warnings(self, user_reputation: Any) -> List[str]:
        """Generate warning messages based on user reputation."""
        warnings = []
        
        if user_reputation.warning_issued and not user_reputation.notification_issued:
            warnings.append(
                f"Warning: This user has shared {user_reputation.false_count} "
                f"false posts out of {user_reputation.total_posts_checked} total posts."
            )
        
        if user_reputation.notification_issued:
            warnings.append(
                f"Alert: This user has a high rate of misinformation. "
                f"{user_reputation.false_count} false posts out of "
                f"{user_reputation.total_posts_checked} total posts."
            )
        
        return warnings

    async def _fallback_analysis(
        self,
        image_bytes: bytes,
        user_prompt: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Fallback analysis when primary verification fails.
        Provides basic analysis with limited functionality.
        """
        try:
            from app.image_processing import text_extractor

            # Basic OCR text extraction
            extracted_elements = text_extractor.extract_social_media_elements(image_bytes)

            # Simple analysis without LLM
            nickname = extracted_elements.get("username", "unknown")
            extracted_text = extracted_elements.get("raw_text", "")

            # Get user reputation
            user_reputation = await self._get_user_reputation(db, nickname)

            # Basic verdict (conservative approach)
            verdict = "partially_true"  # Conservative default
            justification = (
                "Analysis completed with limited functionality. "
                "Primary AI services were unavailable. "
                "This is a conservative assessment - manual verification recommended."
            )

            # Save basic verification result
            verification_record = await self._save_verification_result(
                db, image_bytes, user_prompt,
                {
                    "nickname": nickname,
                    "extracted_text": extracted_text,
                    "primary_topic": "general",
                    "claims": []
                },
                {
                    "verdict": verdict,
                    "justification": justification,
                    "confidence_score": 25,  # Low confidence
                    "sources": ["fallback_analysis"]
                }
            )

            return {
                "status": "partial_success",
                "message": "Analysis completed with limited functionality",
                "verification_id": str(verification_record.id),  # Convert to string for consistency
                "user_nickname": nickname,
                "extracted_text": extracted_text,
                "primary_topic": "general",
                "identified_claims": [],
                "verdict": verdict,
                "justification": justification,
                "confidence_score": 25,
                "user_reputation": {
                    "nickname": user_reputation.nickname,
                    "true_count": user_reputation.true_count,
                    "partially_true_count": user_reputation.partially_true_count,
                    "false_count": user_reputation.false_count,
                    "ironic_count": user_reputation.ironic_count,
                    "total_posts_checked": user_reputation.total_posts_checked,
                    "warning_issued": user_reputation.warning_issued,
                    "notification_issued": user_reputation.notification_issued
                },
                "warnings": [
                    "Analysis completed with limited functionality",
                    "Manual verification recommended",
                    "AI services were temporarily unavailable"
                ]
            }

        except Exception as e:
            logger.error(f"Fallback analysis failed: {e}")
            raise AgentError(f"Both primary and fallback analysis failed: {e}")


# Global agent instance
veritas_agent = VeritasAgent()
