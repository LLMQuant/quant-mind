"""Storage systems for QuantMind knowledge base."""

from quantmind.storage.base import BaseStorage
from quantmind.storage.local_storage import LocalStorage

__all__ = ["BaseStorage", "LocalStorage"]
