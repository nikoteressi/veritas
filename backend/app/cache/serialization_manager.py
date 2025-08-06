"""
Unified serialization manager with automatic algorithm selection.

Provides optimal serialization based on data type and size,
with compression support for large objects.
"""
import json
import logging
import pickle
import sys
import zlib
from typing import Any

import msgpack

from app.exceptions import CacheError

logger = logging.getLogger(__name__)


class SerializationManager:
    """
    Intelligent serialization with automatic algorithm selection.

    Features:
    - Automatic serializer selection based on data type and size
    - Compression for large objects
    - Performance optimization
    - Fallback mechanisms
    """

    def __init__(self, compression_threshold: int = 1024, compression_level: int = 6):
        """
        Initialize serialization manager.

        Args:
            compression_threshold: Size threshold for compression (bytes)
            compression_level: Compression level (1-9)
        """
        self.compression_threshold = compression_threshold
        self.compression_level = compression_level

        # Serialization strategies
        self.strategies = {
            'msgpack_compressed': self._msgpack_compressed,
            'msgpack': self._msgpack,
            'json': self._json,
            'pickle': self._pickle
        }

        # Deserialization strategies
        self.deserializers = {
            b'mpc': self._deserialize_msgpack_compressed,
            b'mp': self._deserialize_msgpack,
            b'js': self._deserialize_json,
            b'pk': self._deserialize_pickle
        }

    def serialize(self, data: Any, hint: str = None) -> bytes:
        """
        Serialize data with optimal algorithm selection.

        Args:
            data: Data to serialize
            hint: Optional hint for serializer selection

        Returns:
            Serialized data with format prefix
        """
        try:
            # Choose optimal serializer
            strategy = self._choose_serializer(data, hint)

            # Serialize with chosen strategy
            serialized_data = self.strategies[strategy](data)

            # Add format prefix for deserialization
            prefix = self._get_prefix(strategy)
            return prefix + serialized_data

        except Exception as e:
            raise CacheError(
                f"Serialization failed with optimal strategy: {e}") from e

    def deserialize(self, data: bytes) -> Any:
        """
        Deserialize data based on format prefix.

        Args:
            data: Serialized data with format prefix

        Returns:
            Deserialized object
        """
        if len(data) < 2:
            raise ValueError("Invalid serialized data: too short")

        # Extract format prefix
        prefix = data[:2]
        payload = data[2:]

        # Get deserializer
        deserializer = self.deserializers.get(prefix)
        if deserializer is None:
            raise ValueError(f"Unknown serialization format: {prefix}")

        return deserializer(payload)

    def _choose_serializer(self, data: Any, hint: str = None) -> str:
        """Choose optimal serializer based on data characteristics."""
        # Use hint if provided
        if hint in self.strategies:
            return hint

        # Calculate data size
        data_size = sys.getsizeof(data)

        # For large objects, use compression
        if data_size > self.compression_threshold:
            return 'msgpack_compressed'

        # For medium objects, use msgpack
        if data_size > 100:
            return 'msgpack'

        # For simple objects, try JSON first
        if self._is_json_serializable(data):
            return 'json'

        # Default to msgpack
        return 'msgpack'

    def _is_json_serializable(self, data: Any) -> bool:
        """Check if data can be serialized with JSON."""
        try:
            json.dumps(data)
            return True
        except (TypeError, ValueError):
            return False

    def _get_prefix(self, strategy: str) -> bytes:
        """Get format prefix for strategy."""
        prefixes = {
            'msgpack_compressed': b'mpc',
            'msgpack': b'mp',
            'json': b'js',
            'pickle': b'pk'
        }
        return prefixes[strategy]

    # Serialization methods
    def _msgpack_compressed(self, data: Any) -> bytes:
        """Serialize with msgpack and compression."""
        msgpack_data = msgpack.packb(data, use_bin_type=True)
        return zlib.compress(msgpack_data, level=self.compression_level)

    def _msgpack(self, data: Any) -> bytes:
        """Serialize with msgpack."""
        return msgpack.packb(data, use_bin_type=True)

    def _json(self, data: Any) -> bytes:
        """Serialize with JSON."""
        return json.dumps(data, separators=(',', ':')).encode('utf-8')

    def _pickle(self, data: Any) -> bytes:
        """Serialize with pickle."""
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

    # Deserialization methods
    def _deserialize_msgpack_compressed(self, data: bytes) -> Any:
        """Deserialize compressed msgpack data."""
        decompressed = zlib.decompress(data)
        return msgpack.unpackb(decompressed, raw=False)

    def _deserialize_msgpack(self, data: bytes) -> Any:
        """Deserialize msgpack data."""
        return msgpack.unpackb(data, raw=False)

    def _deserialize_json(self, data: bytes) -> Any:
        """Deserialize JSON data."""
        return json.loads(data.decode('utf-8'))

    def _deserialize_pickle(self, data: bytes) -> Any:
        """Deserialize pickle data."""
        return pickle.loads(data)

    def get_compression_stats(self, original_data: Any) -> dict:
        """Get compression statistics for data."""
        original_size = len(pickle.dumps(original_data))

        stats = {}
        for strategy_name, strategy_func in self.strategies.items():
            try:
                serialized = strategy_func(original_data)
                compressed_size = len(serialized)
                compression_ratio = original_size / \
                    compressed_size if compressed_size > 0 else 1.0

                stats[strategy_name] = {
                    'size': compressed_size,
                    'compression_ratio': compression_ratio,
                    'space_saved': original_size - compressed_size
                }
            except Exception as e:
                stats[strategy_name] = {'error': str(e)}
                raise CacheError(
                    f"Error serializing with {strategy_name}: {e}") from e

        return stats
