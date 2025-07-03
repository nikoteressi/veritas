"""
Input validation utilities for the Veritas application.
"""
import re
import logging
from io import BytesIO
from typing import Optional, List, Dict, Any
from fastapi import UploadFile
from PIL import Image

from app.exceptions import ValidationError, ImageProcessingError

logger = logging.getLogger(__name__)


class FileValidator:
    """Validator for uploaded files."""
    
    ALLOWED_MIME_TYPES = {
        'image/jpeg',
        'image/jpg', 
        'image/png',
        'image/gif',
        'image/webp'
    }
    
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MIN_FILE_SIZE = 1024  # 1KB
    
    @classmethod
    def validate_image_file(cls, file: UploadFile, image_bytes: bytes) -> None:
        """
        Validate an uploaded image file.
        
        Args:
            file: Uploaded file to validate
            image_bytes: Raw image bytes for content validation
            
        Raises:
            ValidationError: If validation fails
            ImageProcessingError: If the image content is invalid
        """
        # Check if file exists
        if not file:
            raise ValidationError("No file provided")
        
        # Check filename
        if not file.filename:
            raise ValidationError("File must have a filename")
        
        # Check file extension
        file_ext = cls._get_file_extension(file.filename)
        if file_ext not in cls.ALLOWED_EXTENSIONS:
            raise ValidationError(
                f"Invalid file extension. Allowed: {', '.join(cls.ALLOWED_EXTENSIONS)}",
                error_code="INVALID_FILE_EXTENSION",
                details={"allowed_extensions": list(cls.ALLOWED_EXTENSIONS)}
            )
        
        # Check MIME type
        if file.content_type not in cls.ALLOWED_MIME_TYPES:
            raise ValidationError(
                f"Invalid file type. Allowed: {', '.join(cls.ALLOWED_MIME_TYPES)}",
                error_code="INVALID_MIME_TYPE",
                details={"allowed_types": list(cls.ALLOWED_MIME_TYPES)}
            )
        
        # Check file size (if available)
        if hasattr(file, 'size') and file.size:
            if file.size > cls.MAX_FILE_SIZE:
                raise ValidationError(
                    f"File too large. Maximum size: {cls.MAX_FILE_SIZE // (1024*1024)}MB",
                    error_code="FILE_TOO_LARGE",
                    details={"max_size_mb": cls.MAX_FILE_SIZE // (1024*1024)}
                )
            
            if file.size < cls.MIN_FILE_SIZE:
                raise ValidationError(
                    f"File too small. Minimum size: {cls.MIN_FILE_SIZE}B",
                    error_code="FILE_TOO_SMALL",
                    details={"min_size_bytes": cls.MIN_FILE_SIZE}
                )
        
        # Validate image content
        if not cls.is_valid_image_content(image_bytes):
            raise ImageProcessingError(
                "Invalid or corrupted image file",
                error_code="INVALID_IMAGE"
            )
    
    @staticmethod
    def is_valid_image_content(image_bytes: bytes) -> bool:
        """
        Validate that the bytes represent a valid image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            True if valid image, False otherwise
        """
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                img.verify()
            return True
        except Exception as e:
            logger.error(f"Image content validation failed: {e}")
            return False
    
    @staticmethod
    def _get_file_extension(filename: str) -> str:
        """Get file extension from filename."""
        return '.' + filename.split('.')[-1].lower() if '.' in filename else ''


class TextValidator:
    """Validator for text inputs."""
    
    MIN_PROMPT_LENGTH = 5
    MAX_PROMPT_LENGTH = 1000
    
    # Patterns for potentially harmful content
    SUSPICIOUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',  # JavaScript URLs
        r'on\w+\s*=',  # Event handlers
        r'<iframe[^>]*>.*?</iframe>',  # Iframes
    ]
    
    @classmethod
    def validate_prompt(cls, prompt: str) -> str:
        """
        Validate and sanitize a user prompt.
        
        Args:
            prompt: User prompt to validate
            
        Returns:
            Sanitized prompt
            
        Raises:
            ValidationError: If validation fails
        """
        if not prompt:
            raise ValidationError("Prompt cannot be empty")
        
        # Strip whitespace
        prompt = prompt.strip()
        
        # Check length
        if len(prompt) < cls.MIN_PROMPT_LENGTH:
            raise ValidationError(
                f"Prompt too short. Minimum length: {cls.MIN_PROMPT_LENGTH} characters",
                error_code="PROMPT_TOO_SHORT",
                details={"min_length": cls.MIN_PROMPT_LENGTH}
            )
        
        if len(prompt) > cls.MAX_PROMPT_LENGTH:
            raise ValidationError(
                f"Prompt too long. Maximum length: {cls.MAX_PROMPT_LENGTH} characters",
                error_code="PROMPT_TOO_LONG",
                details={"max_length": cls.MAX_PROMPT_LENGTH}
            )
        
        # Check for suspicious content
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                raise ValidationError(
                    "Prompt contains potentially harmful content",
                    error_code="SUSPICIOUS_CONTENT"
                )
        
        # Basic HTML entity encoding for safety
        prompt = cls._sanitize_html(prompt)
        
        return prompt
    
    @classmethod
    def validate_nickname(cls, nickname: str) -> str:
        """
        Validate and sanitize a user nickname.
        
        Args:
            nickname: User nickname to validate
            
        Returns:
            Sanitized nickname
            
        Raises:
            ValidationError: If validation fails
        """
        if not nickname:
            raise ValidationError("Nickname cannot be empty")
        
        # Strip whitespace
        nickname = nickname.strip()
        
        # Check length
        if len(nickname) < 1:
            raise ValidationError("Nickname cannot be empty")
        
        if len(nickname) > 50:
            raise ValidationError(
                "Nickname too long. Maximum length: 50 characters",
                error_code="NICKNAME_TOO_LONG"
            )
        
        # Check for valid characters (alphanumeric, underscore, hyphen)
        if not re.match(r'^[a-zA-Z0-9_-]+$', nickname):
            raise ValidationError(
                "Nickname can only contain letters, numbers, underscores, and hyphens",
                error_code="INVALID_NICKNAME_CHARACTERS"
            )
        
        return nickname
    
    @staticmethod
    def _sanitize_html(text: str) -> str:
        """Basic HTML sanitization."""
        html_entities = {
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '&': '&amp;'
        }
        
        for char, entity in html_entities.items():
            text = text.replace(char, entity)
        
        return text


class RequestValidator:
    """Validator for API requests."""
    
    @staticmethod
    async def validate_verification_request(
        file: UploadFile, 
        prompt: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate a verification request.
        
        Args:
            file: Uploaded image file
            prompt: User prompt
            session_id: Optional WebSocket session ID
            
        Returns:
            Validated and sanitized request data
            
        Raises:
            ValidationError: If validation fails
        """
        # Read image bytes for content validation
        image_bytes = await file.read()
        await file.seek(0)  # Reset file pointer after reading

        # Validate file metadata and content
        FileValidator.validate_image_file(file, image_bytes)
        
        # Validate prompt
        sanitized_prompt = TextValidator.validate_prompt(prompt)
        
        # Validate session ID if provided
        if session_id:
            if not re.match(r'^[a-f0-9-]{36}$', session_id):
                raise ValidationError(
                    "Invalid session ID format",
                    error_code="INVALID_SESSION_ID"
                )
        
        return {
            "file": file,
            "prompt": sanitized_prompt,
            "session_id": session_id,
            "image_bytes": image_bytes
        }
    
    @staticmethod
    def validate_reputation_request(nickname: str) -> str:
        """
        Validate a reputation request.
        
        Args:
            nickname: User nickname
            
        Returns:
            Sanitized nickname
            
        Raises:
            ValidationError: If validation fails
        """
        return TextValidator.validate_nickname(nickname)


# Convenience functions
def validate_image_file(file: UploadFile) -> None:
    """Validate an image file."""
    FileValidator.validate_image_file(file)


def validate_prompt(prompt: str) -> str:
    """Validate and sanitize a prompt."""
    return TextValidator.validate_prompt(prompt)


def validate_nickname(nickname: str) -> str:
    """Validate and sanitize a nickname."""
    return TextValidator.validate_nickname(nickname)
