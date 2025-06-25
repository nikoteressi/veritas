"""
LLM configuration and utilities for Ollama integration.
"""
import base64
import logging
from io import BytesIO
from typing import Optional, Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


class OllamaLLMManager:
    """Manager for Ollama LLM interactions."""
    
    def __init__(self):
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the Ollama LLM."""
        try:
            self.llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0.1,  # Low temperature for factual accuracy
                num_predict=2048,  # Max tokens to generate
                timeout=120,  # 2 minute timeout
            )
            logger.info(f"Initialized Ollama LLM: {settings.ollama_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}")
            raise
    
    def encode_image_to_base64(self, image_bytes: bytes) -> str:
        """
        Encode image bytes to base64 string for multimodal input.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Open image with PIL to validate and potentially resize
            image = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large (optional optimization)
            max_size = (1024, 1024)
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized image from original size to {image.size}")
            
            # Convert back to bytes
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            processed_bytes = buffer.getvalue()
            
            # Encode to base64
            base64_string = base64.b64encode(processed_bytes).decode('utf-8')
            logger.info(f"Encoded image to base64 (size: {len(base64_string)} chars)")
            
            return base64_string
            
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            raise
    
    def create_multimodal_message(self, text: str, image_base64: str) -> HumanMessage:
        """
        Create a multimodal message with text and image.
        
        Args:
            text: Text prompt
            image_base64: Base64 encoded image
            
        Returns:
            HumanMessage with multimodal content
        """
        content = [
            {
                "type": "text",
                "text": text
            },
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_base64}"
            }
        ]
        
        return HumanMessage(content=content)
    
    async def invoke_multimodal(
        self, 
        text: str, 
        image_bytes: bytes,
        **kwargs
    ) -> str:
        """
        Invoke the LLM with multimodal input.
        
        Args:
            text: Text prompt
            image_bytes: Image bytes
            **kwargs: Additional arguments for LLM
            
        Returns:
            LLM response text
        """
        try:
            # Encode image
            image_base64 = self.encode_image_to_base64(image_bytes)
            
            # Create multimodal message
            message = self.create_multimodal_message(text, image_base64)
            
            # Invoke LLM
            response = await self.llm.ainvoke([message], **kwargs)
            
            logger.info("Successfully invoked multimodal LLM")
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to invoke multimodal LLM: {e}")
            raise
    
    async def invoke_text_only(self, text: str, **kwargs) -> str:
        """
        Invoke the LLM with text-only input.
        
        Args:
            text: Text prompt
            **kwargs: Additional arguments for LLM
            
        Returns:
            LLM response text
        """
        try:
            response = await self.llm.ainvoke([HumanMessage(content=text)], **kwargs)
            logger.info("Successfully invoked text-only LLM")
            return response.content
        except Exception as e:
            logger.error(f"Failed to invoke text-only LLM: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "base_url": settings.ollama_base_url,
            "model": settings.ollama_model,
            "status": "initialized" if self.llm else "not_initialized"
        }


# Global LLM manager instance
llm_manager = OllamaLLMManager()
