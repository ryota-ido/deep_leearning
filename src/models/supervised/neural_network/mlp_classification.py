"""MLP classification model implementation using scikit-learn."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.neural_network import MLPClassifier

from src.models.base_model import BaseModel
from src.utils.registry import register_model


@register_model("mlp_classification")
class MLPClassificationModel(BaseModel):
    """
    Multi-layer Perceptron classification model implementation using scikit-learn.

    Provides a wrapper around sklearn's MLPClassifier with
    standard machine learning model interface.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MLP classification model.

        Args:
            config: Configuration dictionary with model parameters.
                   Allows customization of MLPClassifier initialization.
        """
        super().__init__(config)
        self.model_category = "classification"

    def init_model(self):
        """
        Initialize the underlying sklearn MLPClassifier model.

        Uses parameters from self.get_params to configure the model,
        allowing flexible model initialization.
        """
        # Set default parameters if not provided
        params = self.get_params()

        # Handle hidden_layer_sizes specially to allow simpler configuration
        if "hidden_layers" in params and "hidden_layer_sizes" not in params:
            params["hidden_layer_sizes"] = params.pop("hidden_layers")

        # Default parameters if not specified
        if "hidden_layer_sizes" not in params:
            params["hidden_layer_sizes"] = (100, 50)
        if "activation" not in params:
            params["activation"] = "relu"
        if "max_iter" not in params:
            params["max_iter"] = 1000
        if "early_stopping" not in params:
            params["early_stopping"] = True

        self._model = MLPClassifier(**params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> "MLPClassificationModel":
        """
        Fit the MLP classification model to the training data.

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
        Generate predictions using the trained MLP classification model.

        Args:
            X: Input feature matrix of shape (n_samples, n_features)

        Returns:
            Predicted class labels of shape (n_samples,)

        Raises:
            ValueError: If the model has not been fitted before prediction
        """
        self._validate_fitted()
        X = np.asarray(X)
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate class probabilities using the trained MLP model.

        Args:
            X: Input feature matrix of shape (n_samples, n_features)

        Returns:
            Predicted class probabilities of shape (n_samples, n_classes)

        Raises:
            ValueError: If the model has not been fitted before prediction
        """
        self._validate_fitted()
        X = np.asarray(X)
        return self._model.predict_proba(X)
