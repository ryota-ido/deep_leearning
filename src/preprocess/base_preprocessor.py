"""Base preprocessor module."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class BasePreprocessor(ABC):
    """Abstract base class for all data preprocessors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Configuration dictionary for the preprocessor
        """
        self.config = config or {}
        self.fitted = False

    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame,
        target: Optional[np.ndarray] = None,
    ) -> "BasePreprocessor":
        """
        Fit the preprocessor to the data.

        Args:
            data: DataFrame containing the features
            target: Optional target variable

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to the data.

        Args:
            data: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        pass

    def fit_transform(
        self,
        data: pd.DataFrame,
        target: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Args:
            data: DataFrame containing the features
            target: Optional target variable

        Returns:
            Transformed DataFrame
        """
        return self.fit(data, target).transform(data)

    def _validate_fitted(self) -> None:
        """Verify that the preprocessor has been fitted before transforming."""
        if not self.fitted:
            raise ValueError("This preprocessor instance is not fitted yet. " "Call 'fit' before using this method.")

    def _detect_column_types(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Utility method to automatically detect column types from a DataFrame.

        Args:
            data: DataFrame to analyze

        Returns:
            Dictionary containing detected column types grouped by type category
        """
        detected_types = {
            "numeric": data.select_dtypes(include=["int64", "float64"]).columns.tolist(),
            "categorical": data.select_dtypes(include=["object", "category"]).columns.tolist(),
            "datetime": data.select_dtypes(include=["datetime64"]).columns.tolist(),
            "boolean": data.select_dtypes(include=["bool"]).columns.tolist(),
        }
        return detected_types

    def preprocess(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_train_processed = self.fit_transform(X_train, y_train)
        X_test_processed = self.transform(X_test)
        return X_train_processed, X_test_processed
