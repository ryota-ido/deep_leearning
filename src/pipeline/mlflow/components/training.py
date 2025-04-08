"""Model training component for machine learning pipeline."""

import os
from typing import Any, Dict, Optional

import mlflow
import numpy as np

from src.data.manager import DataManager
from src.evaluation.evaluator import Evaluator
from src.models.base_model import BaseModel
from src.pipeline.mlflow.components.base_component import component
from src.training.cross_validation.cross_validation import CrossValidator
from src.training.trainer import Trainer
from src.utils.config_manager import ConfigManager
from src.utils.registry import model_registry


@component("training")
def run_train(config_manager: ConfigManager) -> Dict[str, Any]:
    """
    Train a model based on configuration.

    Args:
        config_manager: Configuration manager instance

    Returns:
        Dictionary containing execution status and results

    Raises:
        ValueError: If required configuration is missing
        FileNotFoundError: If data files are not found
    """
    # Get configuration
    model_config = config_manager.get_model_config()
    data_config = config_manager.get_data_config()
    evaluation_config = config_manager.get_evaluation_config()
    cv_config = config_manager.get_cross_validation_config()

    # Get paths
    processed_dir = config_manager.get_output_path("data", "processed")
    tuned_model_dir = config_manager.get_output_path("models", "tuned")
    model_dir = config_manager.get_output_path("models", "saved")
    os.makedirs(model_dir, exist_ok=True)

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

    # Check if tuned model exists and load it if available
    best_model_path = os.path.join(tuned_model_dir, "best_model.pkl")

    if os.path.exists(best_model_path):
        mlflow.log_param("training.used_tuned_model", True)

        # Load tuned model
        model_type = model_config.get("type", "linear_regression")
        model_registry.discover_modules(base_package="src.models", base_class=BaseModel)
        model_class = model_registry.get(model_type)
        model = model_class.load(best_model_path)
    else:
        mlflow.log_param("training.used_tuned_model", False)

        # Create new model instance
        model_type = model_config.get("type", "linear_regression")
        model_registry.discover_modules(base_package="src.models", base_class=BaseModel)
        model_class = model_registry.get(model_type)
        model = model_class(model_config)

    # Create evaluator and trainer
    evaluator = Evaluator(model.model_category, evaluation_config)
    trainer = Trainer(model, evaluator)

    # Determine whether to use cross-validation
    if cv_config:
        mlflow.log_param("training.with_cv", True)

        # Run cross-validation
        cv = CrossValidator(trainer, cv_config)
        cv_results = cv.cross_validate(X_train_np, y_train_np)

        # Log cross-validation results
        log_cv_info(cv, cv_results, cv_config)

        # Update trainer with best model from CV
        trainer = cv.trainer
    else:
        mlflow.log_param("training.with_cv", False)

        # Train model without cross-validation
        trainer.train(X_train_np, y_train_np, reset=True)

        # Log training information
        log_training_info(trainer)

    # Save trained model
    model_path = os.path.join(model_dir, "trained_model.pkl")
    trainer.save_model(model_path)

    return {"status": "success", "model_path": model_path}


def log_training_info(trainer: Trainer) -> None:
    """
    Log training information to MLflow.

    Args:
        trainer: Trained trainer instance
    """

    # Log model parameters
    model_params = trainer.model.get_params()

    for param_name, param_value in model_params.items():
        if isinstance(param_value, (str, int, float, bool)):
            mlflow.log_param(f"model.{param_name}", param_value)


def log_cv_info(cv: CrossValidator, cv_results: Dict[str, Any], cv_config: Dict[str, Any]) -> None:
    """
    Log cross-validation information to MLflow.

    Args:
        cv: Cross-validator instance
        cv_results: Results from cross-validation
        cv_config: Cross-validation configuration
    """
    # Log cross-validation configuration
    mlflow.log_param("cv.method", cv_config.get("split_method", "unknown"))
    mlflow.log_param("cv.n_splits", cv_config.get("n_splits", 5))
    mlflow.log_param("cv.shuffle", cv_config.get("shuffle", True))
    mlflow.log_param("cv.random_state", cv_config.get("random_state", 42))

    # Log cross-validation scores
    fold_scores = cv_results.get("fold_scores", [])
    if fold_scores:
        mlflow.log_metric("cv.score_mean", np.mean(fold_scores))
        mlflow.log_metric("cv.score_std", np.std(fold_scores))
        mlflow.log_metric("cv.score_min", np.min(fold_scores))
        mlflow.log_metric("cv.score_max", np.max(fold_scores))

        # Log individual fold scores
        for i, score in enumerate(fold_scores):
            mlflow.log_metric(f"cv.fold_{i+1}_score", score)
