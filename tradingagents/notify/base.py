"""Abstract base class for all notifiers."""

from abc import ABC, abstractmethod


class BaseNotifier(ABC):
    """Base class that every notifier must inherit from."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def send(self, message: str) -> bool:
        """Send notification. Returns True on success."""
        ...
