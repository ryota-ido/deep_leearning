"""Hyperparameter tuning component for machine learning pipeline."""

import json
import os
from typing import Any, Dict, Optional

import mlflow
import numpy as np

from src.data.manager import DataManager
from src.evaluation.evaluator import Evaluator
from src.models.base_model import BaseModel
from src.pipeline.mlflow.components.base_component import component
from src.training.trainer import Trainer
from src.training.tuners.base_tuner import BaseTuner
from src.utils.config_manager import ConfigManager
from src.utils.registry import model_registry, tuner_registry


@component("tuning")
def run_tuning(config_manager: ConfigManager) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning if configured.

    Args:
        config_manager: Configuration manager instance

    Returns:
        Dictionary containing execution status and results

    Raises:
        ValueError: If required configuration is missing
        FileNotFoundError: If data files are not found
    """
    # Get configuration
    tuning_config = config_manager.get_tuning_config()
    model_config = config_manager.get_model_config()
    data_config = config_manager.get_data_config()
    evaluation_config = config_manager.get_evaluation_config()

    # Check if tuning is enabled
    if not tuning_config or not tuning_config.get("enabled", False):
        mlflow.log_param("tuning.enabled", False)
        return {"status": "skipped", "reason": "Tuning disabled in config"}

    mlflow.log_param("tuning.enabled", True)
    mlflow.log_param("tuning.type", tuning_config.get("tuner_type", "unknown"))

    # Get output paths
    processed_dir = config_manager.get_output_path("data", "processed")
    tuned_model_dir = config_manager.get_output_path("models", "tuned")
    best_params_dir = config_manager.get_output_path("tuning", "best_params")

    os.makedirs(tuned_model_dir, exist_ok=True)
    os.makedirs(best_params_dir, exist_ok=True)

    # Create data manager
    data_manager = DataManager(data_config)

    # Load training data
    try:
        X_train, y_train = data_manager.load_data(processed_dir, ["X_train.pkl", "y_train.pkl"])
    except FileNotFoundError:
        raise FileNotFoundError("Processed data not found. Run preprocess component first.")

    # Convert DataFrame to numpy if needed
    X_train_np = X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train

    # Extract y values (ensure they're arrays, not DataFrames)
    if hasattr(y_train, "values"):
        y_train_np = y_train.values.flatten()
    else:
        y_train_np = y_train.flatten() if hasattr(y_train, "flatten") else y_train

    # Create model instance
    model_type = model_config.get("type", "linear_regression")
    model_registry.discover_modules(base_package="src.models", base_class=BaseModel)
    model_class = model_registry.get(model_type)
    model = model_class(model_config)

    # Create evaluator and trainer
    evaluator = Evaluator(model.model_category, evaluation_config)
    trainer = Trainer(model, evaluator)

    # Create and run tuner
    tuner_registry.discover_modules(base_package="src.training.tuners", base_class=BaseTuner)
    tuner_type = tuning_config.get("tuner_type", "grid_search")
    tuner_class = tuner_registry.get(tuner_type)
    tuner = tuner_class(trainer, tuning_config)

    # Run tuning
    tuning_results = tuner.tune(X_train_np, y_train_np)

    # Apply best parameters
    best_trainer = tuner.apply_best_params()

    # Save best model
    best_model_path = os.path.join(tuned_model_dir, f"{model_type}_tuned.pkl")
    best_trainer.save_model(best_model_path)

    # Also save as best_model.pkl for standard reference
    standard_model_path = os.path.join(tuned_model_dir, "best_model.pkl")
    best_trainer.save_model(standard_model_path)

    # Save best parameters as JSON
    best_params_path = os.path.join(best_params_dir, f"{model_type}_best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(tuning_results.get("best_params", {}), f, indent=2)

    # Log tuning information
    log_tuning_info(tuner, tuning_results, tuning_config)

    return {"status": "success", "best_params_path": best_params_path, "tuned_model_path": best_model_path}


def log_tuning_info(tuner: BaseTuner, tuning_results: Dict[str, Any], tuning_config: Dict[str, Any]) -> None:
    """
    Log hyperparameter tuning information to MLflow.

    Args:
        tuner: Tuner instance that performed the tuning
        tuning_results: Results from tuning process
        tuning_config: Tuning configuration dictionary
    """
    # Log best score
    if "best_score" in tuning_results:
        mlflow.log_metric("tuning.best_score", tuning_results["best_score"])

    # Log parameter space
    param_grid = tuning_config.get("params", {})
    for param_name, param_values in param_grid.items():
        if isinstance(param_values, list) and not any(isinstance(v, list) for v in param_values):
            # For grid search, log option counts and ranges
            mlflow.log_param(f"tuning.grid.{param_name}.options", len(param_values))
            if all(isinstance(val, (int, float)) for val in param_values):
                mlflow.log_param(f"tuning.grid.{param_name}.min", min(param_values))
                mlflow.log_param(f"tuning.grid.{param_name}.max", max(param_values))
        elif isinstance(param_values, list) and len(param_values) > 0 and param_values[0] == "suggest_float":
            # For Optuna, log parameter range
            if len(param_values) > 1 and isinstance(param_values[1], list) and len(param_values[1]) >= 2:
                mlflow.log_param(f"tuning.optuna.{param_name}.min", param_values[1][0])
                mlflow.log_param(f"tuning.optuna.{param_name}.max", param_values[1][1])

    # Log best parameters
    best_params = tuning_results.get("best_params", {})
    for param_name, param_value in best_params.items():
        mlflow.log_param(f"tuning.best.{param_name}", param_value)

    # Log tuning execution parameters
    if "n_trials" in tuning_config:
        mlflow.log_param("tuning.n_trials", tuning_config["n_trials"])
    if "timeout" in tuning_config:
        mlflow.log_param("tuning.timeout", tuning_config["timeout"])
    if "direction" in tuning_config:
        mlflow.log_param("tuning.direction", tuning_config["direction"])
