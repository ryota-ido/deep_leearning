"""Base tuner module for hyperparameter optimization."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from src.training.cross_validation.cross_validation import CrossValidator
from src.training.trainer import Trainer


class BaseTuner(ABC):
    """
    Abstract base class for hyperparameter tuning strategies.

    Provides a standardized interface for hyperparameter optimization
    across different tuning methods.
    """

    def __init__(self, trainer: Trainer, tuning_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base tuner with a trainer and configuration.

        Args:
            trainer: Trainer instance to be tuned
            config: Configuration dictionary for tuning parameters
        """
        self.trainer = trainer
        self.config = tuning_config or {}
        self.best_params = None
        self.best_score = None
        self.results = None

        self.scoring_metric = self.config.get("scoring_metric", None)
        # Load cross-validation configuration if provided
        cv_config = self.config.get("cross_validation", {})
        self.cross_validator = CrossValidator(trainer, cv_config) if cv_config else None

    @abstractmethod
    def tune(self, X: np.ndarray, y: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Abstract method to perform hyperparameter tuning.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            Dictionary containing tuning results, typically including:
            - best_params: Best hyperparameters found
            - best_score: Score of the best hyperparameters
        """
        pass

    def apply_best_params(self) -> Trainer:
        """
        Apply the best found hyperparameters to the model.

        Returns:
            Trainer instance with best parameters applied

        Raises:
            ValueError: If no best parameters have been found
        """
        if self.best_params is None:
            raise ValueError("No best parameters found. Run tune() first.")

        self.trainer.set_params(**self.best_params)
        return self.trainer

    def get_results(self) -> Dict[str, Any]:
        """
        Retrieve detailed tuning results.

        Returns:
            Dictionary containing comprehensive tuning results

        Raises:
            ValueError: If no results are available
        """
        if self.results is None:
            raise ValueError("No results available. Run tune() first.")

        return self.results

    def _validate_config(self) -> None:
        """
        Validate the tuning configuration.

        Performs basic checks on the configuration to ensure
        it contains necessary parameters for tuning.

        Raises:
            ValueError: If configuration is invalid or incomplete
        """
        # Subclasses can override this method to add specific validation
        if not self.config:
            raise ValueError("Tuning configuration is empty.")
