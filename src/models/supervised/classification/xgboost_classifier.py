import numpy as np
from xgboost import XGBClassifier

from src.models.base_model import BaseModel
from src.utils.registry import register_model


@register_model("xgboost_classifier")
class XGBoostClassifierModel(BaseModel):
    """
    XGBoost Classifier model implementation.

    Args:
        config: Dictionary containing model parameters
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.model_category = "classification"

    def init_model(self):
        """
        Initialize the XGBoost model with parameters from config.
        """
        # Default parameters
        default_params = {
            "estimators_": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 42,
            "use_label_encoder": False,
            "objective": "binary:logistic",
        }

        # Update with user-provided parameters
        params = {**default_params, **self.get_params()}

        # Initialize the model
        self._model = XGBClassifier(**params)

    def fit(self, X, y, **kwargs):
        """
        Train the XGBoost model.

        Args:
            X: Input features
            y: Target values
            **kwargs: Additional parameters to pass to the fit method

        Returns:
            self: The trained model instance
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # For XGBoost, handle eval_set if present in kwargs
        if "eval_set" in kwargs:
            eval_set = kwargs["eval_set"]
            new_eval_set = [(np.asarray(x), np.asarray(y)) for x, y in eval_set]
            kwargs["eval_set"] = new_eval_set

        self._model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Make predictions using the trained model.

        Args:
            X: Input features

        Returns:
            np.ndarray: Predicted values

        Raises:
            ValueError: If the model is not fitted
        """
        self._validate_fitted()
        X = np.asarray(X)
        return self._model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            np.ndarray: Probability estimates for each class

        Raises:
            ValueError: If the model is not fitted
        """
        self._validate_fitted()
        X = np.asarray(X)
        return self._model.predict_proba(X)

    def get_feature_importances(self):
        """
        Get feature importances.

        Returns:
            np.ndarray: Feature importances

        Raises:
            ValueError: If the model is not fitted
        """
        self._validate_fitted()
        return self._model.feature_importances_
