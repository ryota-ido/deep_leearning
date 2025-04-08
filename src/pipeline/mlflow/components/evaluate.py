"""Model evaluation component for machine learning pipeline."""

import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.data.manager import DataManager
from src.evaluation.evaluator import Evaluator
from src.models.base_model import BaseModel
from src.pipeline.mlflow.components.base_component import component
from src.utils.config_manager import ConfigManager
from src.utils.registry import model_registry


@component("evaluate")
def run_evaluate(config_manager: ConfigManager) -> Dict[str, Any]:
    """
    Evaluate a trained model based on configuration.

    Args:
        config_manager: Configuration manager instance

    Returns:
        Dictionary containing execution status and results

    Raises:
        ValueError: If required configuration is missing
        FileNotFoundError: If data files or model are not found
    """
    # Get configuration
    model_config = config_manager.get_model_config()
    data_config = config_manager.get_data_config()
    evaluation_config = config_manager.get_evaluation_config()

    # Get paths
    processed_dir = config_manager.get_output_path("data", "processed")
    model_dir = config_manager.get_output_path("models", "saved")
    results_dir = config_manager.get_output_path("results")
    os.makedirs(results_dir, exist_ok=True)

    # Create data manager
    data_manager = DataManager(data_config)

    # Load test data
    try:
        X_test, y_test = data_manager.load_data(processed_dir, ["X_test.pkl", "y_test.pkl"])
    except FileNotFoundError:
        raise FileNotFoundError("Processed test data not found. Run preprocess component first.")

    # Convert DataFrame to numpy if needed
    X_test_np = X_test.to_numpy() if hasattr(X_test, "to_numpy") else X_test

    # Extract y values (ensure they're arrays, not DataFrames)
    if hasattr(y_test, "values"):
        y_test_np = y_test.values.flatten()
    else:
        y_test_np = y_test.flatten() if hasattr(y_test, "flatten") else y_test

    # Load trained model
    # Load trained model
    model_path = os.path.join(model_dir, "trained_model.pkl")
    if not os.path.exists(model_path):
        # Try best model if trained model is not available
        model_path = os.path.join(config_manager.get_output_path("models", "tuned"), "best_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError("No trained or tuned model found. Run training component first.")

    # Load model
    model_type = model_config.get("type", "linear_regression")
    model_registry.discover_modules(base_package="src.models", base_class=BaseModel)
    model_class = model_registry.get(model_type)
    model = model_class.load(model_path)

    # Create evaluator
    evaluator = Evaluator(model.model_category, evaluation_config)

    # Evaluate model
    metrics = evaluator.evaluate(model, X_test_np, y_test_np)

    # Save evaluation results
    results_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(results_path, "w") as f:
        import json

        json.dump(metrics, f, indent=2)

    # Log evaluation metrics
    log_evaluation_metrics(metrics)

    # Generate and log visualizations if appropriate
    try:
        if model.model_category == "classification":
            generate_classification_visualizations(model, X_test_np, y_test_np, config_manager)
        elif model.model_category == "regression":
            generate_regression_visualizations(model, X_test_np, y_test_np, config_manager)
    except Exception as e:
        mlflow.log_param("visualization_error", str(e))

    return {"status": "success", "metrics": metrics}


def log_evaluation_metrics(metrics: Dict[str, float]) -> None:
    """
    Log evaluation metrics to MLflow.

    Args:
        metrics: Dictionary of evaluation metrics
    """
    # Log each metric to MLflow
    for metric_name, metric_value in metrics.items():
        # Log with eval prefix for organization
        mlflow.log_metric(f"eval.{metric_name}", metric_value)

        # Also log primary metrics at top level for easier tracking
        primary_metrics = {"r2": "r2", "accuracy": "accuracy_score"}
        for key, primary_name in primary_metrics.items():
            if metric_name == primary_name or metric_name == key:
                mlflow.log_metric(key, metric_value)


def generate_classification_visualizations(
    model: BaseModel, X_test: np.ndarray, y_test: np.ndarray, config_manager: ConfigManager
) -> None:
    """
    Generate visualizations for classification models.

    Args:
        model: Trained model to visualize
        X_test: Test features
        y_test: True test labels
        config_manager: Configuration manager instance
    """
    # Get predictions
    y_pred = model.predict(X_test)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create temporary directory for visualizations
    viz_dir = os.path.join(config_manager.get_output_path("results"), "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")

    # Save and log confusion matrix
    cm_path = os.path.join(viz_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # Log as MLflow artifact
    mlflow.log_artifact(cm_path, "evaluation")


def generate_regression_visualizations(
    model: BaseModel, X_test: np.ndarray, y_test: np.ndarray, config_manager: ConfigManager
) -> None:
    """
    Generate visualizations for regression models.

    Args:
        model: Trained model to visualize
        X_test: Test features
        y_test: True test values
        config_manager: Configuration manager instance
    """
    # Get predictions
    y_pred = model.predict(X_test)

    # Create temporary directory for visualizations
    viz_dir = os.path.join(config_manager.get_output_path("results"), "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Plot actual vs predicted
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)

    # Add perfect prediction line
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--")

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted Values")

    # Save and log scatter plot
    scatter_path = os.path.join(viz_dir, "actual_vs_predicted.png")
    plt.savefig(scatter_path)
    plt.close()

    # Log as MLflow artifact
    mlflow.log_artifact(scatter_path, "evaluation")

    # Plot residuals
    plt.figure(figsize=(10, 8))
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    # Save and log residual plot
    residuals_path = os.path.join(viz_dir, "residuals.png")
    plt.savefig(residuals_path)
    plt.close()

    # Log as MLflow artifact
    mlflow.log_artifact(residuals_path, "evaluation")
