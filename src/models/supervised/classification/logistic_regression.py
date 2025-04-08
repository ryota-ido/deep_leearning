"""Logistic regression models implementation."""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from src.models.base_model import BaseModel
from src.utils.registry import register_model


@register_model("logistic_regression")
class LogisticRegression(BaseModel):
    """
    Logistic regression model implementation using scikit-learn.

    Provides a wrapper around sklearn's LogisticRegression with
    standard machine learning model interface.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize logistic regression model.

        Args:
            config: Configuration dictionary with model parameters.
                   Allows customization of LogisticRegression initialization.
        """
        super().__init__(config)
        self.model_category = "classification"

    def init_model(self):
        """
        Initialize the underlying sklearn LogisticRegression model.

        Uses parameters from self.get_params to configure the model,
        allowing flexible model initialization.
        """
        self._model = SklearnLogisticRegression(**self.get_params())

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> "LogisticRegression":
        """
        Fit the logistic regression model to the training data.

        Args:
            X: Training feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            **kwargs: Additional arguments passed to sklearn's fit method

        Returns:
            Self-reference to the model for method chaining
        """
        # Ensure input is numpy array
        X = np.asarray(X)
        y = np.asarray(y)

        # Fit the model
        self._model.fit(X, y, **kwargs)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained logistic regression model.

        Args:
            X: Input feature matrix of shape (n_samples, n_features)

        Returns:
            Predicted target values of shape (n_samples,)

        Raises:
            ValueError: If the model has not been fitted before prediction
        """
        self._validate_fitted()

        return self._model.predict(X)
