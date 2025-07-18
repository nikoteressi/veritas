"""
JSON utilities for handling datetime serialization and other custom types.
"""
import json
import logging
import re
from datetime import datetime, date, time
from decimal import Decimal
from typing import Any, Optional, Dict
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


def json_dumps(obj: Any, **kwargs) -> str:
    """
    Serialize an object to JSON string with support for custom types.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments to pass to json.dumps
        
    Returns:
        JSON string representation
    """
    # Use our custom encoder by default
    if 'cls' not in kwargs:
        kwargs['cls'] = DateTimeJSONEncoder
    
    return json.dumps(obj, **kwargs)


def json_loads(json_str: str, **kwargs) -> Any:
    """
    Deserialize a JSON string to Python object.
    
    Args:
        json_str: JSON string to deserialize
        **kwargs: Additional arguments to pass to json.loads
        
    Returns:
        Deserialized Python object
    """
    return json.loads(json_str, **kwargs)


def prepare_for_json_serialization(obj: Any) -> Any:
    """
    Recursively prepare an object for JSON serialization by converting
    non-serializable types to serializable ones.

    Args:
        obj: Object to prepare

    Returns:
        Object with all non-serializable types converted
    """
    if hasattr(obj, '__dict__'):
        # Handle SQLAlchemy models and other custom objects
        d = {key: prepare_for_json_serialization(value)
             for key, value in obj.__dict__.items()
             if not key.startswith('_')}
        return d
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
