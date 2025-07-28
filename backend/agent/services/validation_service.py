"""
from __future__ import annotations

Centralized validation service for all verification requests.
"""

import logging
from typing import Any

from app.config import settings
from app.exceptions import ValidationError

logger = logging.getLogger(__name__)


class ValidationService:
    """
    Centralized service for handling all validation logic.

    This service encapsulates all validation rules and logic to ensure
    consistent validation across the application.
    """

    def __init__(self):
        # Use the unified configuration from app.config
        self.settings = settings

    def validate_verification_request(
        self,
        file_data: bytes,
        prompt: str,
        filename: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Validate a complete verification request.

        Args:
            file_data: Image file data
            prompt: User prompt
            filename: Optional filename
            session_id: Optional session ID

        Returns:
            Dict containing validated data

        Raises:
            ValidationError: If validation fails
        """
        validated_data = {}

        # Validate image data
        validated_data["image_data"] = self._validate_image_data(file_data, filename)

        # Validate prompt
        validated_data["prompt"] = self._validate_prompt(prompt)

        # Validate session ID if provided
        if session_id:
            validated_data["session_id"] = self._validate_session_id(session_id)

        logger.info(f"Validation successful for request: {filename or 'unknown'}")
        return validated_data

    def _validate_image_data(self, file_data: bytes, filename: str | None) -> bytes:
        """
        Validate image file data.

        Args:
            file_data: Image file data
            filename: Optional filename for logging

        Returns:
            Validated image data

        Raises:
            ValidationError: If image validation fails
        """
        if not file_data:
            raise ValidationError("No image data provided")

        if len(file_data) > self.settings.max_file_size:
            raise ValidationError(
                f"Image too large. Maximum size is {self.settings.max_file_size / (1024 * 1024):.1f}MB"
            )

        if len(file_data) < self.settings.min_file_size:
            raise ValidationError("Image data appears to be corrupted or too small")

        # Basic image format validation (check for common image headers)
        if not self._has_valid_image_signature(file_data):
            raise ValidationError(
                "Invalid image format. Please upload a valid image file."
            )

        return file_data

    def _validate_prompt(self, prompt: str) -> str:
        """
        Validate user prompt.

        Args:
            prompt: User prompt string

        Returns:
            Validated and cleaned prompt

        Raises:
            ValidationError: If prompt validation fails
        """
        if not prompt:
            raise ValidationError("Prompt is required")

        # Clean and normalize prompt
        cleaned_prompt = prompt.strip()

        if len(cleaned_prompt) < self.settings.min_prompt_length:
            raise ValidationError(
                f"Prompt too short. Minimum length is {self.settings.min_prompt_length} characters"
            )

        if len(cleaned_prompt) > self.settings.max_prompt_length:
            raise ValidationError(
                f"Prompt too long. Maximum length is {self.settings.max_prompt_length} characters"
            )

        # Check for obvious spam or malicious content
        if self._contains_suspicious_content(cleaned_prompt):
            raise ValidationError("Prompt contains suspicious content")

        return cleaned_prompt

    def _validate_session_id(self, session_id: str) -> str:
        """
        Validate WebSocket session ID.

        Args:
            session_id: Session ID string

        Returns:
            Validated session ID

        Raises:
            ValidationError: If session ID validation fails
        """
        if not session_id or not session_id.strip():
            raise ValidationError("Session ID cannot be empty")

        cleaned_session_id = session_id.strip()

        # Allow special session IDs for internal use
        special_session_ids = {"sync", "background", "test"}
        if cleaned_session_id in special_session_ids:
            return cleaned_session_id

        # Basic format validation for regular session IDs
        if len(cleaned_session_id) < 8 or len(cleaned_session_id) > 100:
            raise ValidationError("Invalid session ID format")

        return cleaned_session_id

    def _has_valid_image_signature(self, file_data: bytes) -> bool:
        """
        Check if file data has a valid image signature.

        Args:
            file_data: Image file data

        Returns:
            True if valid image signature found
        """
        if len(file_data) < 8:
            return False

        # Check for common image signatures
        image_signatures = [
            b"\xff\xd8\xff",  # JPEG
            b"\x89PNG\r\n\x1a\n",  # PNG
            b"GIF87a",  # GIF87a
            b"GIF89a",  # GIF89a
            b"RIFF",  # WEBP (starts with RIFF)
        ]

        for signature in image_signatures:
            if file_data.startswith(signature):
                return True

        return False

    def _contains_suspicious_content(self, prompt: str) -> bool:
        """
        Check if prompt contains suspicious content.

        Args:
            prompt: User prompt

        Returns:
            True if suspicious content detected
        """
        prompt_lower = prompt.lower()
        return any(
            pattern in prompt_lower for pattern in self.settings.suspicious_patterns
        )

    def validate_analysis_result(
        self, analysis_result: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Validate analysis result structure and content.

        Args:
            analysis_result: Analysis result to validate

        Returns:
            Validated analysis result

        Raises:
            ValidationError: If validation fails
        """
        if not analysis_result:
            raise ValidationError("Analysis result cannot be empty")

        # Validate required fields
        required_fields = ["username", "content", "timestamp"]
        for field in required_fields:
            if field not in analysis_result:
                logger.warning(f"Missing field '{field}' in analysis result")
                analysis_result[field] = None

        # Validate username
        if analysis_result["username"] and len(analysis_result["username"]) > 100:
            raise ValidationError("Username too long")

        # Validate content
        if analysis_result["content"] and len(analysis_result["content"]) > 10000:
            raise ValidationError("Content too long")

        return analysis_result

    def validate_reputation_data(
        self, reputation_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Validate reputation data structure and values.

        Args:
            reputation_data: Reputation data to validate

        Returns:
            Validated reputation data

        Raises:
            ValidationError: If validation fails
        """
        if not reputation_data:
            raise ValidationError("Reputation data cannot be empty")

        # Validate reputation score
        if "reputation_score" in reputation_data:
            score = reputation_data["reputation_score"]
            if not isinstance(score, (int, float)):
                raise ValidationError("Reputation score must be a number")

        return reputation_data


# Singleton instance
validation_service = ValidationService()
