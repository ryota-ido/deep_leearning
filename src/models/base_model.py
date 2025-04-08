"""Base model module for machine learning models."""

import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.

    Provides a standardized interface for model initialization,
    training, prediction, and serialization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base model with configuration.

        Args:
            config (dict, optional): Configuration dictionary for the model.
                Defaults to an empty dictionary if not provided.
        """
        self.config = config or {}
        self.params = self.config.get("params", {})
        self.model_category = "base"
        self.is_fitted = False
        self._model = None

        # Initialize the model state
        self.reset()

    def reset(self):
        """
        Reset the model to its initial state.

        Clears the fitted status and reinitializes the underlying model,
        preparing it for a new training process.
        """
        self.is_fitted = False
        self._model = None
        # Call init_model to set up the initial model configuration
        self.init_model()

    @property
    def model(self):
        """
        Provide controlled access to the underlying model.

        Returns:
            The model instance, initializing it if not already done.
        """
        if self._model is None:
            self.init_model()
        return self._model

    @model.setter
    def model(self, value):
        """
        Set the model with controlled state management.

        Args:
            value: The model instance to set.
        """
        self._model = value
        # Optionally set fitted status when model is set
        # Uncomment if appropriate for specific use cases
        # self.is_fitted = True

    @abstractmethod
    def init_model(self):
        """
        Abstract method to initialize the specific model.

        Subclasses must implement this method to create
        their specific model instance with appropriate configurations.
        """
        pass

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        **kwargs,
    ) -> "BaseModel":
        """
        Train the model on the given training data.

        Args:
            X: Training feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            **kwargs: Additional keyword arguments for model-specific training

        Returns:
            Self-reference to the model for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for the input data.

        Args:
            X: Input feature matrix of shape (n_samples, n_features)

        Returns:
            Predicted values of shape (n_samples,)

        Raises:
            ValueError: If the model has not been fitted before prediction
        """
        pass

    def save(self, filepath: str) -> None:
        """
        Save the model to a file using pickle serialization.

        Args:
            filepath: Path where the model will be saved

        Raises:
            ValueError: If the model has not been fitted before saving
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "BaseModel":
        """
        Load a previously saved model from a file.

        Args:
            filepath: Path to the saved model file

        Returns:
            Loaded model instance

        Raises:
            TypeError: If the loaded model is not an instance of the expected class
        """
        with open(filepath, "rb") as f:
            model = pickle.load(f)

        if not isinstance(model, cls):
            raise TypeError(f"Loaded model is not an instance of {cls.__name__}")

        return model

    def _validate_fitted(self) -> None:
        """
        Validate that the model has been fitted before prediction.

        Raises:
            ValueError: If the model is not fitted or the internal model is None
        """
        if not self.is_fitted or self._model is None:
            raise ValueError("This model instance is not fitted yet. Call 'fit' before using this method.")

    def get_params(self):
        """
        Retrieve model parameters.
        Modified to safely handle pre-initialized models.

        Returns:
            dict: Model parameters
        """
        # Prioritize explicitly specified parameters in configuration
        if self.config and "params" in self.config:
            return self.config.get("params", {})

        # Return empty dictionary if model is not initialized
        if not hasattr(self, "_model") or self._model is None:
            return {}

        # Call get_params only if model is initialized
        try:
            # Pre-check if specific attribute exists
            if hasattr(self._model, "get_params"):
                return self._model.get_params()
        except Exception:
            # Return empty dictionary if get_params call raises an exception
            return {}

        return {}

    def set_params(self, **params) -> "BaseModel":
        """
        Set new parameters for the model.

        Args:
            **params: Keyword arguments representing model parameters to update

        Returns:
            Self-reference to the model for method chaining
        """
        if self._model and hasattr(self._model, "set_params"):
            self._model.set_params(**params)
            self.params = params

        return self
