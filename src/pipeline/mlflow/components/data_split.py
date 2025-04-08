"""Data splitting component for machine learning pipeline."""

import os
from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd

from src.data.manager import DataManager
from src.pipeline.mlflow.components.base_component import component
from src.utils.config_manager import ConfigManager


@component("data_split")
def run_data_split(config_manager: ConfigManager) -> Dict[str, Any]:
    """
    Split data into training and testing sets based on configuration.

    Args:
        config_manager: Configuration manager instance

    Returns:
        Dictionary containing execution status and results

    Raises:
        ValueError: If required configuration is missing
        FileNotFoundError: If data file is not found
    """
    # Get data configuration
    data_config = config_manager.get_data_config()

    # Get output paths
    raw_dir = config_manager.get_output_path("data", "raw")
    split_dir = config_manager.get_output_path("data", "split")
    os.makedirs(split_dir, exist_ok=True)

    # Create data manager
    data_manager = DataManager(data_config)

    # Load data from previous step
    try:
        data = data_manager.load_data(raw_dir, "dataset.pkl")
    except FileNotFoundError:
        raise FileNotFoundError("Dataset not found. Run data_loader component first.")

    # Get split parameters
    target_col = data_config.get("target_col", "target")
    test_size = data_config.get("split", {}).get("test_size", 0.2)
    random_state = data_config.get("split", {}).get("random_state", 42)
    stratify = data_config.get("split", {}).get("stratify", False)

    # Split data
    X_train, X_test, y_train, y_test = data_manager.split_data(
        data=data, target_col=target_col, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Ensure y_train and y_test are DataFrame objects for consistent saving
    if not isinstance(y_train, pd.DataFrame):
        y_train = pd.DataFrame(y_train, columns=["target"])

    if not isinstance(y_test, pd.DataFrame):
        y_test = pd.DataFrame(y_test, columns=["target"])

    # Save split results
    data_manager.save_data(
        [X_train, X_test, y_train, y_test], ["X_train.pkl", "X_test.pkl", "y_train.pkl", "y_test.pkl"], split_dir
    )

    # Log split information
    log_split_info(X_train, X_test, y_train, y_test, target_col, test_size, random_state, stratify)

    return {"status": "success"}


def log_split_info(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
    stratify: bool,
) -> None:
    """
    Log data split information to MLflow.

    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training target values
        y_test: Testing target values
        target_col: Target column name
        test_size: Test size ratio
        random_state: Random seed for reproducibility
        stratify: Whether stratified sampling was used
    """
    # Log split parameters
    mlflow.log_param("split.target_column", target_col)
    mlflow.log_metric("split.test_size_pct", test_size * 100)
    mlflow.log_param("split.random_state", random_state)
    mlflow.log_param("split.stratify", stratify)

    # Log split results
    mlflow.log_metric("split.train_samples", X_train.shape[0])
    mlflow.log_metric("split.test_samples", X_test.shape[0])
    mlflow.log_metric("split.train_pct", X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]) * 100)
    mlflow.log_metric("split.test_pct", X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]) * 100)

    # Log feature information
    mlflow.log_param("split.feature_count", X_train.shape[1])

    # Log class distribution if classification task (assuming y has 10 or fewer unique values)
    y_train_values = y_train.values.flatten()
    y_test_values = y_test.values.flatten()

    unique_values = np.unique(np.concatenate([y_train_values, y_test_values]))
    if len(unique_values) <= 10:  # Reasonable threshold for classification
        for cls in unique_values:
            train_count = (y_train_values == cls).sum()
            test_count = (y_test_values == cls).sum()
            mlflow.log_metric(f"split.train_class_{cls}", train_count)
            mlflow.log_metric(f"split.test_class_{cls}", test_count)
