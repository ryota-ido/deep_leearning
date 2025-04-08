"""Data loading component for machine learning pipeline (functional approach)."""

import os
from typing import Any, Dict

import mlflow
import pandas as pd

from src.data.manager import DataManager
from src.pipeline.mlflow.components.base_component import component
from src.utils.config_manager import ConfigManager


@component("data_loader")
def run_data_loader(config_manager: ConfigManager) -> Dict[str, Any]:
    """
    Load data based on configuration.

    Args:
        config_manager: Configuration manager instance

    Returns:
        Dictionary containing execution status and results

    Raises:
        ValueError: If required configuration is missing
        FileNotFoundError: If data file is not found
    """
    # Ensure output directories exist
    for category in ["data", "models", "tuning"]:
        if category == "data":
            for subdir in ["raw", "split", "processed"]:
                os.makedirs(config_manager.get_output_path(category, subdir), exist_ok=True)
        elif category == "models":
            for subdir in ["saved", "tuned"]:
                os.makedirs(config_manager.get_output_path(category, subdir), exist_ok=True)
        elif category == "tuning":
            for subdir in ["best_params", "history"]:
                os.makedirs(config_manager.get_output_path(category, subdir), exist_ok=True)

    os.makedirs(config_manager.get_output_path("logs"), exist_ok=True)
    os.makedirs(config_manager.get_output_path("results"), exist_ok=True)

    # Get data configuration
    data_config = config_manager.get_data_config()

    # Get file information
    file = data_config.get("file")
    if not file:
        raise ValueError("Data file not specified in configuration")

    format = data_config.get("format", "csv")

    # Get output path
    raw_dir = config_manager.get_output_path("data", "raw")

    # Create data manager
    data_manager = DataManager(data_config)

    # Load data
    data = data_manager.load_data("", file, format)

    # Save data
    data_manager.save_data(data, "dataset.pkl", raw_dir)

    # Log data information
    log_data_info(data, file)

    return {"status": "success"}


def log_data_info(data: pd.DataFrame, data_file: str) -> None:
    """
    Log data information to MLflow.

    Args:
        data: Loaded data DataFrame
        data_file: Path to the data file
    """
    # Log basic data information
    mlflow.log_param("data.file", data_file)
    mlflow.log_param("data.rows", data.shape[0])
    mlflow.log_param("data.columns", data.shape[1])

    # Log column names (if not too many)
    if data.shape[1] <= 20:
        mlflow.log_param("data.column_names", ", ".join(data.columns.astype(str).tolist()))

    # Log data type distribution
    dtypes = data.dtypes.astype(str).value_counts().to_dict()
    for dtype, count in dtypes.items():
        mlflow.log_param(f"data.dtype.{dtype}", count)

    # Log missing values information
    missing_values = data.isnull().sum().sum()
    mlflow.log_param("data.missing_values.total", missing_values)

    # Log missing values by column (if any)
    missing_by_column = data.isnull().sum()
    missing_columns = missing_by_column[missing_by_column > 0]
    for col, count in missing_columns.items():
        col_name = str(col).replace(".", "_")
        mlflow.log_param(f"data.missing_values.{col_name}", count)
