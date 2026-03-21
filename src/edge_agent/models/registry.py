import logging
import threading
from typing import Dict, Type

from .base import ModelWrapper

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    A thread-safe singleton registry that caches loaded model instances.
    Prevents redundant weight loading across multiple evaluators.
    """

    _instances: Dict[str, ModelWrapper] = {}
    _lock = threading.Lock()

    @classmethod
    def get_model(
        cls, model_class: Type[ModelWrapper], name: str, weights_path: str, **kwargs
    ) -> ModelWrapper:
        """
        Retrieves a cached model instance or creates and loads a new one.
        """
        key = f"{model_class.__name__}:{weights_path}"

        with cls._lock:
            if key not in cls._instances:
                logger.info(
                    f"Registry: Loading new {name} instance from {weights_path}"
                )
                instance = model_class(name, weights_path=weights_path, **kwargs)
                instance.load()
                cls._instances[key] = instance
            else:
                logger.debug(f"Registry: Reusing cached {name} instance")

            return cls._instances[key]

    @classmethod
    def clear(cls):
        """
        Unloads and clears all cached model instances.
        """
        with cls._lock:
            for instance in cls._instances.values():
                instance.unload()
            cls._instances.clear()
            logger.info("Registry: All model instances cleared.")
