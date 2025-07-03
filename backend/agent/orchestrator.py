"""
Core orchestration for Veritas fact-checking.
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
from typing import Dict, Any, AsyncGenerator, Optional, List, Type

from langchain_core.output_parsers import JsonOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession

from agent.llm import llm_manager
from agent.tools import AVAILABLE_TOOLS, searxng_tool
from agent.temporal_analysis import temporal_analyzer
from app.exceptions import AgentError
from app.schemas import ImageAnalysisResult, FactCheckResult, VerdictResult
from agent.services.image_analysis import image_analysis_service
from agent.services.fact_checking import FactCheckingService
from agent.services.verdict import verdict_service
from agent.services.reputation import reputation_service
from agent.services.storage import storage_service

logger = logging.getLogger(__name__)


class VerificationOrchestrator:
    """Orchestrates the verification workflow by calling various services."""
    
    def __init__(self):
        self.llm = llm_manager.llm
        self.tools = AVAILABLE_TOOLS
        self.agent_executor = None
        self._initialize_agent()
        self.fact_checking_service = FactCheckingService(searxng_tool)
    
    def _initialize_agent(self):
        """Initialize the LangChain agent executor."""
        try:
            from langchain_core.prompts import ChatPromptTemplate
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant with access to tools for fact-checking."),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            
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
        session_id: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Main method to verify a social media post.
        """
        start_time = time.time()
        
        try:
            # Step 1: Multimodal Analysis
            if progress_callback:
                await progress_callback("Analyzing image content...", 10)
            analysis_result = await image_analysis_service.analyze(image_bytes, user_prompt)
            extracted_info = analysis_result.dict()
            
            # Step 2: Get user reputation
            if progress_callback:
                await progress_callback("Checking user reputation...", 35)
            user_reputation = await reputation_service.get_or_create(
                db, extracted_info.get("username", "unknown")
            )

            # Step 3: Temporal analysis
            if progress_callback:
                await progress_callback("Analyzing temporal context...", 40)
            temporal_analysis = temporal_analyzer.analyze_temporal_context(extracted_info)
            extracted_info["temporal_analysis"] = temporal_analysis

            # Step 4: Fact-checking
            if progress_callback:
                await progress_callback("Fact-checking claims...", 50)
            fact_check_result = await self.fact_checking_service.check(
                extracted_info, user_prompt, progress_callback
            )
            
            # Step 5: Generate final verdict
            if progress_callback:
                await progress_callback("Generating final verdict...", 80)
            verdict_result = await verdict_service.generate(
                fact_check_result, user_prompt, extracted_info.get("temporal_analysis", {})
            )
            
            # Step 6: Update user reputation
            if progress_callback:
                await progress_callback("Updating user reputation...", 90)
            updated_reputation = await reputation_service.update(
                db, extracted_info.get("username", "unknown"), verdict_result.verdict
            )
            
            # Step 7: Save verification result
            if progress_callback:
                await progress_callback("Saving results...", 95)
            
            reputation_data = {
                "nickname": updated_reputation.nickname,
                "true_count": updated_reputation.true_count,
                "partially_true_count": updated_reputation.partially_true_count,
                "false_count": updated_reputation.false_count,
                "ironic_count": updated_reputation.ironic_count,
                "total_posts_checked": updated_reputation.total_posts_checked,
                "warning_issued": updated_reputation.warning_issued,
                "notification_issued": updated_reputation.notification_issued
            }

            verification_record = await storage_service.save_verification_result(
                db=db,
                user_nickname=updated_reputation.nickname,
                image_bytes=image_bytes, 
                user_prompt=user_prompt, 
                extracted_info=extracted_info, 
                verdict_result=verdict_result.dict(), 
                reputation_data=reputation_data
            )

            # Step 8: Compile final result BEFORE storing in vector DB
            processing_time = int(time.time() - start_time)
            warnings = reputation_service.generate_warnings(updated_reputation)
            
            final_result = {
                "status": "success",
                "message": "Verification completed successfully",
                "verification_id": str(verification_record.id),
                "nickname": extracted_info.get("username"),
                "extracted_text": analysis_result.extracted_text,
                "primary_topic": analysis_result.primary_topic,
                "claims": analysis_result.claims,
                "verdict": verdict_result.verdict,
                "reasoning": verdict_result.reasoning,
                "confidence_score": verdict_result.confidence_score,
                "processing_time_seconds": processing_time,
                "temporal_analysis": extracted_info.get("temporal_analysis", {}),
                "fact_check_results": {
                    "examined_sources": fact_check_result.examined_sources,
                    "search_queries_used": fact_check_result.search_queries_used,
                    "summary": fact_check_result.summary.dict(),
                },
                "user_reputation": reputation_data,
                "warnings": warnings
            }

            # Step 9: Store the compiled result in the vector database
            await storage_service.store_in_vector_db(final_result)
            
            if progress_callback:
                await progress_callback("Verification complete!", 100)
            
            logger.info(f"Verification completed in {processing_time}s: {verdict_result.verdict}")
            return final_result
            
        except Exception as e:
            logger.error(f"Verification failed: {e}", exc_info=True)
            raise AgentError(f"An unexpected error occurred during verification: {e}") from e

# Singleton instance of the main orchestrator
verification_orchestrator = VerificationOrchestrator()
