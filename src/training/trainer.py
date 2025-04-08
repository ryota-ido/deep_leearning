"""Trainer module for model training and evaluation."""

from typing import Any, Dict, Optional

import numpy as np

from src.evaluation.evaluator import Evaluator
from src.models.base_model import BaseModel


class Trainer:
    """
    A comprehensive trainer for machine learning models.

    Handles model training, evaluation, and provides a unified interface
    for model-related operations.
    """

    def __init__(
        self,
        model: BaseModel,
        evaluator: Optional[Evaluator],
    ):
        """
        Initialize the trainer with a model and configuration.

        Args:
            model: Machine learning model to be trained
            evaluation_config: Configuration dictionary for evaluation parameters
        """
        self.model = model
        self.evaluator = evaluator or Evaluator(self.model.model_category, {})

    def train(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        reset: bool = False,
        **kwargs,
    ) -> BaseModel:
        """
        Train the model on the given data.

        Args:
            X: Training feature matrix
            y: Target values (optional for some unsupervised models)
            reset: If True, reinitialize the model before training
            **kwargs: Additional keyword arguments for model training

        Returns:
            Trained model instance
        """
        if reset:
            self.model.init_model()

        # Train the model
        self.model.fit(X, y, **kwargs)
        return self.model

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            X: Test feature matrix
            y: True target values
            custom_metrics: Optional dictionary of custom metrics to use

        Returns:
            Dictionary of evaluation metrics

        Raises:
            ValueError: If the evaluator is not initialized
        """
        if self.evaluator is None:
            raise ValueError("Model must be trained before evaluation")

        return self.evaluator.evaluate(self.model, X, y)

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path where the model will be saved
        """
        self.model.save(filepath)

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model file
        """
        self.model = self.model.__class__.load(filepath)

    def get_params(self) -> Dict[str, Any]:
        """
        Retrieve the current model parameters.

        Returns:
            Dictionary of model parameters
        """
        return self.model.get_params()

    def set_params(self, **params) -> "Trainer":
        """
        Update the model's parameters.

        Args:
            **params: Keyword arguments representing parameters to update

        Returns:
            Self-reference to the trainer for method chaining
        """
        self.model.set_params(**params)
        return self
