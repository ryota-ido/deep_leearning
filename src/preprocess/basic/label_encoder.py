"""Label encoding implementation for categorical data preprocessing."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder

from src.preprocess.base_preprocessor import BasePreprocessor
from src.utils.registry import register_preprocessor


@register_preprocessor("label_encoder")
class LabelEncoder(BasePreprocessor):
    """
    Encode categorical features as integer labels.

    Transforms categorical variables into sequential integers starting from 0.
    Different values from the categories will be encoded as different integers.
    Assumes missing values are already handled before this preprocessor.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        handle_unknown: str = "error",
    ):
        """
        Initialize label encoder for categorical data.

        Args:
            config: Configuration dictionary
            columns: Specific columns to encode (None = all categorical columns)
            handle_unknown: Strategy for handling unknown categories in transform
                            ('error' or 'use_default')
        """
        super().__init__(config)

        # Set from config if provided, otherwise use constructor parameters
        if config:
            self.columns = config.get("columns", columns)
            self.handle_unknown = config.get("handle_unknown", handle_unknown)
            self.default_value = config.get("default_value", -1)
        else:
            self.columns = columns
            self.handle_unknown = handle_unknown
            self.default_value = -1

        # Dictionary to store encoders for each column
        self.encoders: Dict[str, SklearnLabelEncoder] = {}

        # Dictionary to store original categories
        self.categories: Dict[str, List[Any]] = {}

    def fit(self, data, target=None):
        """
        Fit the label encoder to the categorical data.

        Args:
            data: Input DataFrame (missing values should be handled before)
            target: Optional target variable (not used for this preprocessor)

        Returns:
            Self reference for method chaining
        """
        # Determine which columns to process
        if self.columns is None:
            # Get all categorical columns (object, category, string)
            categorical_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            # Use specified columns that exist in the data
            categorical_columns = [col for col in self.columns if col in data.columns]

        # Fit encoder for each categorical column
        for col in categorical_columns:
            # Skip columns with all NaN values
            if data[col].isna().all():
                continue

            # Get non-null values for fitting
            fit_data = data[col].dropna().astype(str)

            # Create and fit the encoder if there's data
            if len(fit_data) > 0:
                encoder = SklearnLabelEncoder()
                encoder.fit(fit_data)
                self.encoders[col] = encoder

                # Store original categories for reference
                self.categories[col] = list(encoder.classes_)

        self.fitted = True
        return self

    def transform(self, data):
        """
        Transform categorical columns to integer labels.

        Args:
            data: Input DataFrame with categorical columns (missing values handled)

        Returns:
            DataFrame with encoded categorical values

        Raises:
            ValueError: If the encoder hasn't been fitted yet
        """
        self._validate_fitted()

        # Create a copy to avoid modifying the original data
        result = data.copy()

        # Transform each column that has an encoder
        for col, encoder in self.encoders.items():
            if col not in result.columns:
                continue

            # Create a mask for non-missing values
            mask_valid = result[col].notna()

            if mask_valid.any():
                # Convert valid values to string type for encoding
                valid_values = result.loc[mask_valid, col].astype(str)

                # Check for unknown categories if needed
                if self.handle_unknown == "use_default":
                    # Use default value for unknown categories
                    for idx, val in zip(valid_values.index, valid_values.values):
                        if val in self.categories[col]:
                            result.loc[idx, col] = encoder.transform([val])[0]
                        else:
                            result.loc[idx, col] = self.default_value
                else:
                    # Let scikit-learn's LabelEncoder handle errors
                    try:
                        result.loc[mask_valid, col] = encoder.transform(valid_values)
                    except ValueError as e:
                        raise ValueError(
                            f"Unknown categories found in column '{col}'. "
                            f"Set handle_unknown to 'use_default' or handle them before encoding. "
                            f"Original error: {str(e)}"
                        )

        return result

    def reset(self) -> "LabelEncoder":
        """
        Reset fitted state and encoders.

        Returns:
            Self reference for method chaining
        """
        self.encoders = {}
        self.categories = {}
        self.fitted = False
        return self

    def get_categories(self, column: str) -> Optional[List[Any]]:
        """
        Get categories for a specific column.

        Args:
            column: Column name

        Returns:
            List of categories or None if column wasn't encoded

        Raises:
            ValueError: If the encoder hasn't been fitted yet
        """
        self._validate_fitted()
        return self.categories.get(column)
