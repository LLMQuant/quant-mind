"""Storage systems for QuantMind knowledge base."""

from quantmind.storage.base import BaseStorage
from quantmind.storage.json_storage import JSONStorage

__all__ = ["BaseStorage", "JSONStorage"]
