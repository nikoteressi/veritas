"""
JSON utilities for handling datetime serialization and other custom types.
"""
import json
import logging
from datetime import datetime, date, time
from decimal import Decimal
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)


class DateTimeJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles datetime objects and other common types
    that are not natively JSON serializable.
    """
    
    def default(self, obj: Any) -> Any:
        """
        Convert non-serializable objects to JSON-serializable format.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable representation of the object
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, time):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, UUID):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            # Handle custom objects by converting to dict
            return obj.__dict__
        else:
            # Let the base class handle other types
            return super().default(obj)


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely serialize an object to JSON string with datetime support.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments to pass to json.dumps
        
    Returns:
        JSON string representation
    """
    try:
        # Use our custom encoder by default
        if 'cls' not in kwargs:
            kwargs['cls'] = DateTimeJSONEncoder
        
        return json.dumps(obj, **kwargs)
    except Exception as e:
        logger.error(f"Failed to serialize object to JSON: {e}")
        # Return a safe fallback
        return json.dumps({"error": "serialization_failed", "message": str(e)})


def safe_json_loads(json_str: str, **kwargs) -> Any:
    """
    Safely deserialize a JSON string to Python object.
    
    Args:
        json_str: JSON string to deserialize
        **kwargs: Additional arguments to pass to json.loads
        
    Returns:
        Deserialized Python object
    """
    try:
        return json.loads(json_str, **kwargs)
    except Exception as e:
        logger.error(f"Failed to deserialize JSON string: {e}")
        return {"error": "deserialization_failed", "message": str(e)}


def prepare_for_json_serialization(obj: Any) -> Any:
    """
    Recursively prepare an object for JSON serialization by converting
    non-serializable types to serializable ones.
    
    Args:
        obj: Object to prepare
        
    Returns:
        Object with all non-serializable types converted
    """
    if isinstance(obj, dict):
        return {key: prepare_for_json_serialization(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [prepare_for_json_serialization(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(prepare_for_json_serialization(item) for item in obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, date):
        return obj.isoformat()
    elif isinstance(obj, time):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, UUID):
        return str(obj)
    else:
        return obj


def create_serializable_dict(data: dict) -> dict:
    """
    Create a dictionary that is guaranteed to be JSON serializable.
    
    Args:
        data: Dictionary to make serializable
        
    Returns:
        JSON-serializable dictionary
    """
    return prepare_for_json_serialization(data)
