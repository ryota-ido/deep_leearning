"""Outlier detection and removal implementations for data preprocessing."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.preprocess.base_preprocessor import BasePreprocessor
from src.utils.registry import register_preprocessor


@register_preprocessor("outlier_remover")
class OutlierRemover(BasePreprocessor):
    """
    Detects and handles outliers in numerical data.

    Provides multiple detection methods including IQR (Interquartile Range),
    Z-score, and custom threshold-based approaches.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        method: str = "iqr",
        threshold: float = 1.5,
        treatment: str = "clip",
        columns: Optional[List[str]] = None,
        z_threshold: float = 3.0,
    ):
        """
        Initialize outlier detector and remover.

        Args:
            config: Configuration dictionary
            method: Outlier detection method ('iqr', 'zscore', 'threshold')
            threshold: Threshold multiplier for IQR method or absolute value for threshold method
            treatment: How to handle outliers ('clip', 'remove', 'null')
            columns: Specific columns to check for outliers (None = all numeric columns)
            z_threshold: Z-score threshold when using 'zscore' method
        """
        super().__init__(config)

        # Set from config if provided, otherwise use constructor parameters
        if config:
            self.method = config.get("method", method)
            self.threshold = config.get("threshold", threshold)
            self.treatment = config.get("treatment", treatment)
            self.columns = config.get("columns", columns)
            self.z_threshold = config.get("z_threshold", z_threshold)
        else:
            self.method = method
            self.threshold = threshold
            self.treatment = treatment
            self.columns = columns
            self.z_threshold = z_threshold

        # Values to be learned during fitting
        self.outlier_bounds: Dict[str, Tuple[float, float]] = {}
        self.column_stats: Dict[str, Dict[str, float]] = {}

    def fit(self, data, target=None):
        """
        Learn outlier boundaries from training data.

        Args:
            data: Input DataFrame
            target: Optional target variable (not used for this preprocessor)

        Returns:
            Self reference for method chaining
        """
        # Determine which columns to process
        if self.columns is None:
            # Get all numeric columns
            columns_to_process = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        else:
            # Use specified columns that exist in the data
            columns_to_process = [col for col in self.columns if col in data.columns]

        for col in columns_to_process:
            # Skip columns with too many NaN values
            if data[col].isna().sum() > len(data) * 0.5:
                continue

            # Calculate bounds based on selected method
            if self.method == "iqr":
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (self.threshold * iqr)
                upper_bound = q3 + (self.threshold * iqr)

                self.outlier_bounds[col] = (lower_bound, upper_bound)
                self.column_stats[col] = {"q1": q1, "q3": q3, "iqr": iqr}

            elif self.method == "zscore":
                mean = data[col].mean()
                std = data[col].std()

                # Avoid division by zero
                if std == 0:
                    continue

                lower_bound = mean - (self.z_threshold * std)
                upper_bound = mean + (self.z_threshold * std)

                self.outlier_bounds[col] = (lower_bound, upper_bound)
                self.column_stats[col] = {"mean": mean, "std": std}

            elif self.method == "threshold":
                min_val = data[col].min()
                max_val = data[col].max()

                self.outlier_bounds[col] = (min_val - self.threshold, max_val + self.threshold)
                self.column_stats[col] = {"min": min_val, "max": max_val}

            else:
                raise ValueError(f"Unknown outlier detection method: {self.method}")

        self.fitted = True
        return self

    def transform(self, data):
        """
        Detect and handle outliers based on learned boundaries.

        Args:
            data: Input DataFrame potentially containing outliers

        Returns:
            DataFrame with outliers handled according to specified treatment

        Raises:
            ValueError: If the remover hasn't been fitted yet
        """
        self._validate_fitted()

        # Create a copy to avoid modifying the original data
        result = data.copy()

        if not self.outlier_bounds:
            # No outlier bounds were learned, return original data
            return result

        # Process each column with learned boundaries
        for col, (lower_bound, upper_bound) in self.outlier_bounds.items():
            if col not in result.columns:
                continue

            # Identify outliers
            outlier_mask = (result[col] < lower_bound) | (result[col] > upper_bound)

            if not outlier_mask.any():
                continue

            # Apply the specified treatment
            if self.treatment == "clip":
                result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)

            elif self.treatment == "null":
                result.loc[outlier_mask, col] = np.nan

            elif self.treatment == "remove":
                # This will drop rows with outliers - be careful with this option!
                result = result[~outlier_mask].reset_index(drop=True)

            else:
                raise ValueError(f"Unknown outlier treatment method: {self.treatment}")

        return result

    def reset(self):
        """
        Reset fitted state and learned parameters.

        Returns:
            Self reference for method chaining
        """
        self.outlier_bounds = {}
        self.column_stats = {}
        self.fitted = False
        return self

    def get_outlier_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about detected outliers.

        Returns:
            Dictionary with column-wise outlier statistics

        Raises:
            ValueError: If the remover hasn't been fitted yet
        """
        self._validate_fitted()

        stats = {}
        for col, bounds in self.outlier_bounds.items():
            stats[col] = {"bounds": bounds, "stats": self.column_stats.get(col, {})}

        return stats
