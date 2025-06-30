"""
Image processing utilities for multimodal analysis.
"""
import hashlib
import logging
from io import BytesIO
from typing import Tuple, Optional, Dict, Any

from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

from app.exceptions import ImageProcessingError

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image processing operations for social media posts."""
    
    def __init__(self):
        self.max_size = (1920, 1080)  # Max resolution for processing
        self.quality = 85  # JPEG quality for compression
    
    def validate_image(self, image_bytes: bytes) -> bool:
        """
        Validate that the uploaded file is a valid image.
        
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
            logger.error(f"Image validation failed: {e}")
            return False
    
    def get_image_info(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Get basic information about the image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dictionary with image information
        """
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                return {
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "width": img.width,
                    "height": img.height,
                    "file_size": len(image_bytes),
                    "has_transparency": img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                }
        except Exception as e:
            logger.error(f"Failed to get image info: {e}")
            return {}
    
    def preprocess_image(self, image_bytes: bytes) -> bytes:
        """
        Preprocess image for better OCR and analysis.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Processed image bytes
        """
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large
                if img.size[0] > self.max_size[0] or img.size[1] > self.max_size[1]:
                    img.thumbnail(self.max_size, Image.Resampling.LANCZOS)
                    logger.info(f"Resized image to {img.size}")
                
                # Enhance image for better text recognition
                img = self._enhance_for_ocr(img)
                
                # Convert back to bytes
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=self.quality, optimize=True)
                processed_bytes = buffer.getvalue()
                
                logger.info(f"Preprocessed image: {len(image_bytes)} -> {len(processed_bytes)} bytes")
                return processed_bytes
                
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}", exc_info=True)
            raise ImageProcessingError(f"Image preprocessing failed: {e}") from e
    
    def _enhance_for_ocr(self, img: Image.Image) -> Image.Image:
        """
        Enhance image specifically for OCR text extraction.
        
        Args:
            img: PIL Image object
            
        Returns:
            Enhanced PIL Image object
        """
        # Increase contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        
        # Increase sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.1)
        
        # Apply slight noise reduction
        img = img.filter(ImageFilter.MedianFilter(size=3))
        
        return img
    
    def extract_text_ocr(self, image_bytes: bytes) -> str:
        """
        Extract text from image using OCR (fallback method).
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Extracted text string
        """
        try:
            # Preprocess image for better OCR
            processed_bytes = self.preprocess_image(image_bytes)
            
            with Image.open(BytesIO(processed_bytes)) as img:
                # Use pytesseract for OCR
                text = pytesseract.image_to_string(img, lang='eng')
                
                # Clean up the text
                text = self._clean_ocr_text(text)
                
                logger.info(f"OCR extracted {len(text)} characters")
                return text
                
        except Exception as e:
            logger.error(f"OCR text extraction failed: {e}", exc_info=True)
            raise ImageProcessingError(f"OCR text extraction failed: {e}") from e
    
    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean up OCR-extracted text.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Join lines with single newlines
        cleaned = '\n'.join(lines)
        
        # Remove excessive spaces
        import re
        cleaned = re.sub(r' +', ' ', cleaned)
        
        return cleaned.strip()
    
    def generate_image_hash(self, image_bytes: bytes) -> str:
        """
        Generate SHA-256 hash of image for deduplication.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(image_bytes).hexdigest()
    
    def create_thumbnail(self, image_bytes: bytes, size: Tuple[int, int] = (200, 200)) -> bytes:
        """
        Create a thumbnail of the image.
        
        Args:
            image_bytes: Raw image bytes
            size: Thumbnail size (width, height)
            
        Returns:
            Thumbnail image bytes
        """
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create thumbnail
                img.thumbnail(size, Image.Resampling.LANCZOS)
                
                # Convert to bytes
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=80)
                
                return buffer.getvalue()
                
        except Exception as e:
            logger.error(f"Thumbnail creation failed: {e}")
            raise ImageProcessingError(f"Thumbnail creation failed: {e}") from e
    
    def detect_image_type(self, image_bytes: bytes) -> str:
        """
        Detect the type of social media post image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Image type classification
        """
        try:
            info = self.get_image_info(image_bytes)
            
            # Simple heuristics for image type detection
            width, height = info.get('size', (0, 0))
            aspect_ratio = width / height if height > 0 else 1
            
            # Common social media aspect ratios
            if 0.9 <= aspect_ratio <= 1.1:
                return "square_post"  # Instagram square, etc.
            elif aspect_ratio > 1.5:
                return "landscape_post"  # Twitter, Facebook landscape
            elif aspect_ratio < 0.7:
                return "story_format"  # Instagram/Facebook story
            else:
                return "standard_post"
                
        except Exception as e:
            logger.error(f"Image type detection failed: {e}")
            return "unknown"


class TextExtractor:
    """Specialized text extraction for social media posts."""
    
    def __init__(self):
        self.image_processor = ImageProcessor()
    
    def extract_social_media_elements(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract social media specific elements from the image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dictionary with extracted elements
        """
        try:
            # Get basic OCR text
            ocr_text = self.image_processor.extract_text_ocr(image_bytes)
            
            # Parse social media elements
            elements = self._parse_social_media_text(ocr_text)
            
            # Add image metadata
            elements.update({
                "image_info": self.image_processor.get_image_info(image_bytes),
                "image_type": self.image_processor.detect_image_type(image_bytes),
                "image_hash": self.image_processor.generate_image_hash(image_bytes)
            })
            
            return elements
            
        except Exception as e:
            logger.error(f"Social media element extraction failed: {e}")
            return {"raw_text": "", "username": "", "timestamp": "", "content": ""}
    
    def _parse_social_media_text(self, text: str) -> Dict[str, Any]:
        """
        Parse OCR text to identify social media elements.
        
        Args:
            text: OCR extracted text
            
        Returns:
            Dictionary with parsed elements
        """
        import re
        
        elements = {
            "raw_text": text,
            "username": "",
            "timestamp": "",
            "content": "",
            "hashtags": [],
            "mentions": [],
            "urls": []
        }
        
        if not text:
            return elements
        
        lines = text.split('\n')
        
        # Look for username patterns (simplified)
        username_patterns = [
            r'@(\w+)',  # @username
            r'(\w+)\s*•',  # username •
            r'^(\w+)$'  # standalone word that might be username
        ]
        
        for line in lines[:3]:  # Check first few lines for username
            for pattern in username_patterns:
                match = re.search(pattern, line)
                if match:
                    elements["username"] = match.group(1)
                    break
            if elements["username"]:
                break
        
        # Look for hashtags
        hashtags = re.findall(r'#(\w+)', text)
        elements["hashtags"] = hashtags
        
        # Look for mentions
        mentions = re.findall(r'@(\w+)', text)
        elements["mentions"] = mentions
        
        # Look for URLs (simplified)
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        elements["urls"] = urls
        
        # Extract main content (everything that's not username/timestamp)
        content_lines = []
        for line in lines:
            # Skip lines that look like usernames or timestamps
            if not (re.match(r'^\w+\s*•', line) or re.match(r'^\d+[hmd]', line)):
                content_lines.append(line)
        
        elements["content"] = '\n'.join(content_lines).strip()
        
        return elements


# Global instances
image_processor = ImageProcessor()
text_extractor = TextExtractor()
