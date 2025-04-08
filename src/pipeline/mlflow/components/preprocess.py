"""Preprocessing component for machine learning pipeline."""

import os
from typing import Any, Dict, List

import mlflow
import numpy as np
import pandas as pd

from src.data.manager import DataManager
from src.pipeline.mlflow.components.base_component import component
from src.preprocess.preprocess_pipeline import PreprocessingPipeline
from src.utils.config_manager import ConfigManager


@component("preprocess")
def run_preprocess(config_manager: ConfigManager) -> Dict[str, Any]:
    """
    Apply preprocessing steps to training and testing data based on configuration.

    Args:
        config_manager: Configuration manager instance

    Returns:
        Dictionary containing execution status and results

    Raises:
        ValueError: If required configuration is missing
        FileNotFoundError: If data files are not found
    """
    # Get configuration
    preprocessing_config = config_manager.get_preprocessing_config()
    data_config = config_manager.get_data_config()

    # Get path information
    split_dir = config_manager.get_output_path("data", "split")
    processed_dir = config_manager.get_output_path("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Create data manager
    data_manager = DataManager(data_config)

    # Load split data
    try:
        X_train, X_test, y_train, y_test = data_manager.load_data(
            split_dir, ["X_train.pkl", "X_test.pkl", "y_train.pkl", "y_test.pkl"]
        )
    except FileNotFoundError:
        raise FileNotFoundError("Split data not found. Run data_split component first.")

    # Extract y values (ensure they're arrays, not DataFrames)
    if isinstance(y_train, pd.DataFrame):
        y_train_values = y_train.values.flatten()
    else:
        y_train_values = y_train

    # Create and run preprocessing pipeline
    preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)
    X_train_processed, X_test_processed = preprocessing_pipeline.run(X_train, y_train_values, X_test)

    # Save processed data
    data_manager.save_data(
        [X_train_processed, X_test_processed, y_train, y_test],
        ["X_train.pkl", "X_test.pkl", "y_train.pkl", "y_test.pkl"],
        processed_dir,
    )

    # Log preprocessing information
    log_preprocessing_info(X_train, X_train_processed, preprocessing_config)

    return {"status": "success"}


def log_preprocessing_info(
    X_train_original: pd.DataFrame, X_train_processed: pd.DataFrame, preprocessing_config: Dict[str, Any]
) -> None:
    """
    Log preprocessing information to MLflow.

    Args:
        X_train_original: Original training features DataFrame
        X_train_processed: Processed training features DataFrame
        preprocessing_config: Configuration dictionary for preprocessing
    """
    # Log basic preprocessing information
    mlflow.log_param("preprocess.original_features", X_train_original.shape[1])
    mlflow.log_param("preprocess.processed_features", X_train_processed.shape[1])

    # Log preprocessing steps
    steps = preprocessing_config.get("steps", [])
    mlflow.log_param("preprocess.steps_count", len(steps))

    for i, step in enumerate(steps):
        step_type = step.get("type", "unknown")
        step_name = step.get("name", f"step_{i}")
        mlflow.log_param(f"preprocess.step_{i}.name", step_name)
        mlflow.log_param(f"preprocess.step_{i}.type", step_type)

        # Log important step parameters
        params = step.get("params", {})
        for param_name, param_value in params.items():
            # Log only scalar parameters
            if isinstance(param_value, (str, int, float, bool)):
                mlflow.log_param(f"preprocess.step_{i}.{param_name}", param_value)

    # Log feature changes
    if X_train_original.shape[1] == X_train_processed.shape[1]:
        # If feature count is the same, detect changed features
        original_cols = set(X_train_original.columns)
        processed_cols = set(X_train_processed.columns)

        added_cols = processed_cols - original_cols
        removed_cols = original_cols - processed_cols

        if added_cols:
            mlflow.log_param("preprocess.added_features", ", ".join(added_cols))
        if removed_cols:
            mlflow.log_param("preprocess.removed_features", ", ".join(removed_cols))
