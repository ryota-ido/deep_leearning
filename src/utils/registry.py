"""Registry pattern implementation for dynamic class management."""

import importlib
import inspect
import logging
import os
import sys
from typing import Callable, Dict, Type, TypeVar

logger = logging.getLogger(__name__)

# Type variable for generic registry
T = TypeVar("T")


class Registry:
    """
    Generic registry for managing class mappings with dynamic discovery.

    Provides a flexible mechanism for registering, retrieving,
    and dynamically discovering classes across modules.
    """

    def __init__(self, name: str):
        """
        Initialize a new registry.

        Args:
            name: Unique identifier for the registry
        """
        self.name = name
        self._registry: Dict[str, Type[T]] = {}

    def register(self, key: str) -> Callable:
        """
        Create a decorator for registering classes in the registry.

        Args:
            key: Unique identifier for the class

        Returns:
            Decorator function for class registration
        """

        def decorator(cls: Type[T]) -> Type[T]:
            """
            Decorator to register a class.

            Args:
                cls: Class to be registered

            Returns:
                The original class, unmodified

            Raises:
                ValueError: If the key is already registered
            """
            if key in self._registry:
                raise ValueError(f"Duplicate registration: {key} already exists " f"in {self.name} registry")
            self._registry[key] = cls
            return cls

        return decorator

    def get(self, key: str) -> Type[T]:
        """
        Retrieve a registered class by its key.

        Args:
            key: Identifier of the class to retrieve

        Returns:
            Registered class

        Raises:
            KeyError: If the key is not found in the registry
        """
        if key not in self._registry:
            available_keys = list(self._registry.keys())
            raise KeyError(f"Key '{key}' not found in {self.name} registry. " f"Available keys: {available_keys}")
        return self._registry[key]

    def get_all(self) -> Dict[str, Type[T]]:
        """
        Retrieve all registered classes.

        Returns:
            Dictionary of all registered classes
        """
        return self._registry.copy()

    def discover_modules(self, base_package: str, base_class: Type):
        """
        Dynamically discover and import modules containing subclasses.

        Args:
            base_package: Base package path to search for classes
            base_class: Base class to identify relevant subclasses
        """
        # Convert package notation to file system path
        base_path = base_package.replace(".", os.sep)

        if not os.path.exists(base_path):
            logger.warning(f"Package path '{base_path}' does not exist.")
            return

        # Import top-level Python files
        for file in os.listdir(base_path):
            if file.endswith(".py") and file != "__init__.py":
                module_name = f"{base_package}.{file[:-3]}"
                try:
                    importlib.import_module(module_name)
                except ModuleNotFoundError:
                    logger.warning(f"Could not import '{module_name}', skipping.")

        # Recursively import modules in subdirectories
        for root, _, files in os.walk(base_path):
            rel_path = os.path.relpath(root, base_path)
            if rel_path == ".":
                continue  # Skip base package

            for file in files:
                if file.endswith(".py") and file != "__init__.py":
                    module_name = ".".join([base_package] + rel_path.split(os.sep) + [file[:-3]])
                    try:
                        importlib.import_module(module_name)
                    except ModuleNotFoundError:
                        logger.warning(f"Could not import '{module_name}', skipping.")

        # Identify and register subclasses
        for module in list(sys.modules.values()):
            if module and module.__name__.startswith(base_package):
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, base_class) and obj is not base_class and obj not in self._registry.values():
                        # Note: This does not automatically register the class
                        # Classes should still be explicitly registered using decorators
                        logger.debug(f"Discovered potential {self.name} class: {obj.__name__}")


# Global registry instances
model_registry = Registry("model")
tuner_registry = Registry("tuner")
preprocessor_registry = Registry("preprocessor")


def register_model(key: str):
    """
    Decorator to register a model class in the model registry.

    Args:
        key: Unique identifier for the model class

    Returns:
        Class registration decorator
    """
    return model_registry.register(key)


def register_tuner(key: str):
    """
    Decorator to register a tuner class in the tuner registry.

    Args:
        key: Unique identifier for the tuner class

    Returns:
        Class registration decorator
    """
    return tuner_registry.register(key)


def register_preprocessor(key: str):
    """
    Decorator to register a preprocessor class in the preprocessor registry.

    Args:
        key: Unique identifier for the preprocessor class

    Returns:
        Class registration decorator
    """
    return preprocessor_registry.register(key)
