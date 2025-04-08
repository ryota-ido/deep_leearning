"""Main experiment execution module for machine learning workflow."""

import logging
import time

from src.pipeline.mlflow.mlflow_pipeline import run_pipeline

# Set up logger
logger = logging.getLogger(__name__)


def run_experiment(config_path: str) -> None:
    """
    Execute a complete machine learning experiment based on configuration.

    Orchestrates the entire machine learning workflow from data loading
    to model evaluation.

    Args:
        config_path: Configuration path dictionary defining experiment parameters
    """

    try:
        logger.info("Starting machine learning experiment...")
        start_time = time.time()

        run_info = run_pipeline(config_path)
        for name, value in run_info.get("metrics", {}).items():
            print(f"{name}: {value:.4f}")
        print(f"Execution time: {run_info['execution_time']:.2f} seconds")

        total_experiment_time = time.time() - start_time
        logger.info(f"Experiment completed successfully in {total_experiment_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise
