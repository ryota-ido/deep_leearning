"""Evaluation module for machine learning model performance assessment."""

import importlib
import logging
from typing import Any, Callable, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

# Common metric functions mapping
COMMON_METRICS: Dict[str, Any] = {
    # Basic metrics
    "accuracy_score": accuracy_score,
    "precision_score": precision_score,
    "recall_score": recall_score,
    "f1_score": f1_score,
    # Regression metrics
    "mean_squared_error": mean_squared_error,
    "mean_absolute_error": mean_absolute_error,
    "r2_score": r2_score,
}

# Default metric parameters by model type
DEFAULT_METRIC_PARAMS: Dict[str, Any] = {
    "classification": {
        "precision": {"average": "weighted", "zero_division": 0},
        "recall": {"average": "weighted", "zero_division": 0},
        "f1_score": {"average": "weighted", "zero_division": 0},
    },
    "regression": {},  # No special parameters for regression metrics
}


class Evaluator:
    """
    Model evaluation utility that calculates performance metrics.

    Provides functionality to evaluate machine learning models using
    configured metrics from settings.
    """

    def __init__(self, model_category: str, evaluation_config: Optional[Dict[str, Any]] = None):
        """
        Initialize evaluator for a specific model type with configuration.

        Args:
            model_category: Type of model to evaluate (e.g., 'regression', 'classification')
            evaluation_config: Configuration dictionary for evaluation settings
        """
        self.model_category = model_category
        self.metrics: Dict[str, Any] = {}
        self.primary_metric = None

        # Initialize with default metrics for the model type
        self._set_default_metrics()

        # Override with custom evaluation configuration if provided
        if evaluation_config:
            self.apply_evaluation_config(evaluation_config)

    def _set_default_metrics(self) -> None:
        """Set up default metrics based on the model type."""
        if self.model_category == "regression":
            self._add_regression_metrics()
        elif self.model_category == "classification":
            self._add_classification_metrics()

    def _add_regression_metrics(self) -> None:
        """Add standard regression metrics to the evaluator."""
        self.metrics["mean_squared_error"] = COMMON_METRICS["mean_squared_error"]
        self.metrics["mean_absolute_error"] = COMMON_METRICS["mean_absolute_error"]
        self.metrics["r2_score"] = COMMON_METRICS["r2_score"]

    def _add_classification_metrics(self) -> None:
        """Add standard classification metrics to the evaluator."""
        self.metrics["accuracy_score"] = COMMON_METRICS["accuracy_score"]

        # Add metrics with specific parameters for classification
        params = DEFAULT_METRIC_PARAMS["classification"]

        # Create bound functions with the appropriate parameters
        self.metrics["precision_score"] = lambda y, y_pred: COMMON_METRICS["precision_score"](
            y, y_pred, **params["precision_score"]
        )
        self.metrics["recall_score"] = lambda y, y_pred: COMMON_METRICS["recall_score"](
            y, y_pred, **params["recall_score"]
        )
        self.metrics["f1_score"] = lambda y, y_pred: COMMON_METRICS["f1_score"](y, y_pred, **params["f1_score"])

    def apply_evaluation_config(self, config: Dict[str, Any]) -> "Evaluator":
        """
        Apply evaluation configuration from a structured dictionary.

        This method configures the evaluator using the structured format:
        {
            "primary_metric": "metric_name",
            "metrics": {
                "metric1": {"enabled": true, "params": {...}},
                "metric2": {"enabled": false},
                ...
            }
        }

        Args:
            config: Evaluation configuration dictionary

        Returns:
            Self-reference for method chaining
        """
        # Set primary metric if specified
        if "primary_metric" in config:
            self.primary_metric = config["primary_metric"]
            logger.info(f"Set primary metric to: {self.primary_metric}")

        # Process structured metrics configuration
        if "metrics" in config and isinstance(config["metrics"], dict):
            metrics_config: Dict[str, Any] = config["metrics"]

            # Clear existing metrics if specified
            if config.get("replace_defaults", False):
                self.metrics = {}
                logger.info("Cleared default metrics as specified in config")

            for metric_name, metric_config in metrics_config.items():
                # Skip disabled metrics
                if metric_config.get("enabled", True) is False:
                    if metric_name in self.metrics:
                        del self.metrics[metric_name]
                        logger.info(f"Disabled metric: {metric_name}")
                    continue

                # Resolve the metric function
                metric_func = self._resolve_metric_function(metric_name)
                if metric_func is None:
                    logger.warning(f"Unknown metric: {metric_name}")
                    continue

                # Apply parameters if specified
                params = metric_config.get("params", {})
                if params:
                    # Create a bound function with the specified parameters
                    bound_func = lambda y, y_pred, f=metric_func, p=params: f(y, y_pred, **p)
                    self.metrics[metric_name] = bound_func
                    logger.info(f"Added metric '{metric_name}' with custom parameters")
                else:
                    self.metrics[metric_name] = metric_func
                    logger.info(f"Added metric '{metric_name}'")

        return self

    def _resolve_metric_function(self, function_reference: str) -> Optional[Callable]:
        """
        Resolve a metric function reference to the actual function.

        Args:
            function_reference: String reference to a metric function

        Returns:
            Resolved metric function or None if resolution fails
        """
        # First check if it's a common metric
        if function_reference in COMMON_METRICS:
            return COMMON_METRICS[function_reference]

        # Otherwise try to dynamically import
        try:
            module_path, function_name = function_reference.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, function_name)
        except (ValueError, ImportError, AttributeError) as e:
            logger.warning(f"Error resolving metric function '{function_reference}': {e}")
            return None

    def evaluate(self, model: BaseModel, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance using configured metrics from settings.

        Args:
            model: Machine learning model to evaluate
            X: Input feature matrix
            y: True target values

        Returns:
            Dictionary of metric names and their corresponding values

        Raises:
            ValueError: If model prediction fails
        """
        if not self.metrics:
            logger.warning("No metrics configured for evaluation")
            return {}

        try:
            # Generate predictions
            y_pred = model.predict(X)

            # Calculate metrics
            results = {}
            for name, metric_func in self.metrics.items():
                try:
                    value = metric_func(y, y_pred)
                    # Ensure the metric value is a scalar
                    if isinstance(value, (np.ndarray, list)):
                        value = float(np.mean(value))
                    results[name] = float(value)
                except Exception as e:
                    error_msg = str(e)
                    results[f"{name}_error"] = error_msg
                    logger.warning(f"Error calculating metric '{name}': {error_msg}")

            return results

        except Exception as e:
            logger.error(f"Evaluation failed during prediction: {e}")
            raise ValueError(f"Model prediction failed during evaluation: {e}")

    def get_primary_metric(self) -> Optional[str]:
        """
        Get the name of the primary metric used for model selection.

        Returns:
            Name of the primary metric or None if not set
        """
        return self.primary_metric

    def custom(self):
        return 0
