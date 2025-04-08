"""Missing value handling implementation for data preprocessing."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.preprocess.base_preprocessor import BasePreprocessor
from src.utils.registry import register_preprocessor


@register_preprocessor("missing_value_handler")
class MissingValueHandler(BasePreprocessor):
    """
    Handles missing values in datasets using various strategies.

    Supports different imputation methods for numerical and categorical data,
    with options for column-specific handling.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        strategy: str = "mean",
        fill_value: Optional[Any] = None,
        categorical_strategy: str = "most_frequent",
        categorical_fill_value: Optional[str] = "MISSING",
        column_strategies: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the missing value handler.

        Args:
            config: Configuration dictionary
            strategy: Default strategy for numerical columns
                      ('mean', 'median', 'constant', 'drop')
            fill_value: Value to use with 'constant' strategy for numerical columns
            categorical_strategy: Strategy for categorical columns
                                 ('most_frequent', 'constant', 'drop')
            categorical_fill_value: Value to use with 'constant' strategy for categorical columns
            column_strategies: Dictionary mapping column names to specific strategies
        """
        super().__init__(config)

        # Set from config if provided, otherwise use constructor parameters
        if config:
            self.strategy = config.get("strategy", strategy)
            self.fill_value = config.get("fill_value", fill_value)
            self.categorical_strategy = config.get("categorical_strategy", categorical_strategy)
            self.categorical_fill_value = config.get("categorical_fill_value", categorical_fill_value)
            self.column_strategies = config.get("column_strategies", column_strategies or {})
        else:
            self.strategy = strategy
            self.fill_value = fill_value
            self.categorical_strategy = categorical_strategy
            self.categorical_fill_value = categorical_fill_value
            self.column_strategies = column_strategies or {}

        # Values learned during fitting
        self.imputation_values: Dict[str, Any] = {}
        self.columns_to_drop: List[str] = []

    def fit(self, data, target=None):
        """
        Learn imputation values from the training data.

        Args:
            data: Input DataFrame with missing values
            target: Optional target variable (not used for this preprocessor)

        Returns:
            Self reference for method chaining
        """
        # Detect column types
        column_types = self._detect_column_types(data)
        numeric_cols = column_types["numeric"]
        categorical_cols = column_types["categorical"]

        # Process each column based on its type
        for col in data.columns:
            # Skip columns with no missing values
            if not data[col].isna().any():
                continue

            # Get column-specific strategy or use default based on type
            col_strategy = self.column_strategies.get(col)

            if col in numeric_cols:
                col_strategy = col_strategy or self.strategy

                if col_strategy == "drop":
                    self.columns_to_drop.append(col)
                elif col_strategy == "mean":
                    self.imputation_values[col] = data[col].mean()
                elif col_strategy == "median":
                    self.imputation_values[col] = data[col].median()
                elif col_strategy == "constant":
                    self.imputation_values[col] = self.fill_value
                else:
                    raise ValueError(f"Unknown strategy '{col_strategy}' for numeric column '{col}'")

            elif col in categorical_cols:
                col_strategy = col_strategy or self.categorical_strategy

                if col_strategy == "drop":
                    self.columns_to_drop.append(col)
                elif col_strategy == "most_frequent":
                    self.imputation_values[col] = data[col].mode()[0] if not data[col].mode().empty else None
                elif col_strategy == "constant":
                    self.imputation_values[col] = self.categorical_fill_value
                else:
                    raise ValueError(f"Unknown strategy '{col_strategy}' for categorical column '{col}'")

        self.fitted = True
        return self

    def transform(self, data):
        """
        Impute missing values based on learned parameters.

        Args:
            data: Input DataFrame with missing values

        Returns:
            DataFrame with missing values handled according to strategies

        Raises:
            ValueError: If the handler hasn't been fitted yet
        """
        self._validate_fitted()

        # Create a copy to avoid modifying the original data
        result = data.copy()

        # Drop columns if specified
        if self.columns_to_drop:
            result = result.drop(columns=self.columns_to_drop, errors="ignore")

        # Impute missing values using learned parameters
        for col, value in self.imputation_values.items():
            if col in result.columns:
                result[col] = result[col].fillna(value)

        return result

    def reset(self):
        """
        Reset fitted state and learned parameters.

        Returns:
            Self reference for method chaining
        """
        self.imputation_values = {}
        self.columns_to_drop = []
        self.fitted = False
        return self
