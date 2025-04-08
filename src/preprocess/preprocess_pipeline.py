"""Preprocessing pipeline component for machine learning workflows."""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.preprocess.base_preprocessor import BasePreprocessor
from src.utils.registry import preprocessor_registry


class PreprocessingPipeline:
    """
    Preprocessing pipeline for applying a sequence of preprocessing steps
    defined via a configuration dictionary.
    """

    def __init__(self, preprocessing_config: Dict[str, Any]) -> None:
        """
        Initialize the preprocessing pipeline.

        Args:
            preprocessing_config: Configuration dictionary containing preprocessing settings
        """
        self.config = preprocessing_config
        self.steps = self._create_steps()

    def _create_steps(self) -> List[Tuple[str, BasePreprocessor]]:
        """
        Create preprocessing steps from configuration.

        Returns:
            List of tuples containing (step_name, preprocessor_instance)
        """
        steps_config = self.config.get("steps", [])
        if not steps_config:
            return []

        preprocessor_registry.discover_modules(base_package="src.preprocess", base_class=BasePreprocessor)

        preprocessing_steps = []
        for i, step_config in enumerate(steps_config):
            step_name = step_config.get("name", f"step_{i}")
            preprocessor = self._create_preprocessor(step_config)
            preprocessing_steps.append((step_name, preprocessor))

        return preprocessing_steps

    def _create_preprocessor(self, preprocessor_config: Dict[str, Any]) -> BasePreprocessor:
        """
        Create a preprocessor instance from configuration.

        Args:
            preprocessor_config: Configuration dictionary for a single preprocessor step

        Returns:
            Initialized preprocessor instance

        Raises:
            ValueError: If the specified preprocessor type is not registered
        """
        preprocessor_type = preprocessor_config.get("type", "")
        params = preprocessor_config.get("params", {})

        preprocessor_class = preprocessor_registry.get(preprocessor_type)
        if preprocessor_class is None:
            raise ValueError(f"Preprocessor '{preprocessor_type}' is not registered.")

        return preprocessor_class(params)

    def run(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply the preprocessing pipeline to training and test data.

        Args:
            X_train: Training feature DataFrame
            y_train: Training target values
            X_test: Test feature DataFrame

        Returns:
            Tuple containing (processed_X_train, processed_X_test)
        """
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()

        for name, preprocessor in self.steps:
            X_train_processed, X_test_processed = preprocessor.preprocess(X_train_processed, X_test_processed, y_train)

        return X_train_processed, X_test_processed
