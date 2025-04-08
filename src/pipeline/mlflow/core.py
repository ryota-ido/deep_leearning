"""Core pipeline implementation for machine learning experiments."""

import logging
import time
from typing import Any, Dict, List, Optional

import mlflow

from src.pipeline.mlflow.components.data_loader import run_data_loader
from src.pipeline.mlflow.components.data_split import run_data_split
from src.pipeline.mlflow.components.evaluate import run_evaluate
from src.pipeline.mlflow.components.preprocess import run_preprocess
from src.pipeline.mlflow.components.training import run_train
from src.pipeline.mlflow.components.tuning import run_tuning
from src.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class MLflowPipeline:
    """
    MLflow-based machine learning pipeline with isolated components.

    Each component receives the configuration manager and handles data
    loading/saving independently, communicating through the file system.
    """

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the MLflow pipeline with configuration.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager

        # MLflow configuration
        mlflow_config = self.config_manager.get_mlflow_config()
        self.experiment_name = mlflow_config.get("experiment_name", "default_experiment")
        self.tracking_uri = mlflow_config.get("tracking_uri", None)
        self.tags = mlflow_config.get("tags", {})
        self.artifact_location = mlflow_config.get("artifact_location", None)
        self.registry_uri = mlflow_config.get("registry_uri", None)

        # List of pipeline components in execution order
        self.components = self._get_pipeline_components()

        # Configure MLflow
        self._setup_mlflow()

    def _get_pipeline_components(self) -> List[Dict[str, Any]]:
        """
        Get list of pipeline components with their configuration.

        Returns:
            List of component definitions
        """
        return [
            {"name": "data_loader", "function": run_data_loader},
            {"name": "data_split", "function": run_data_split},
            {"name": "preprocess", "function": run_preprocess},
            {"name": "tuning", "function": run_tuning},
            {"name": "training", "function": run_train},
            {"name": "evaluation", "function": run_evaluate},
        ]

    def _setup_mlflow(self) -> None:
        """
        Set up MLflow tracking environment.
        """
        try:
            # Set tracking URI if provided
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
                logger.info(f"MLflow tracking URI set to: {self.tracking_uri}")

            # Set registry URI if provided
            if self.registry_uri:
                mlflow.set_registry_uri(self.registry_uri)
                logger.info(f"MLflow registry URI set to: {self.registry_uri}")

            # Get or create the experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)

            if experiment is None:
                # Create experiment with optional artifact location
                experiment_kwargs = {}
                if self.artifact_location:
                    experiment_kwargs["artifact_location"] = self.artifact_location

                self.experiment_id = mlflow.create_experiment(self.experiment_name, **experiment_kwargs)
                logger.info(f"Created new experiment: {self.experiment_name} (ID: {self.experiment_id})")
            else:
                self.experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name} (ID: {self.experiment_id})")

        except Exception as e:
            logger.error(f"Error setting up MLflow: {str(e)}")
            raise

    def _log_config(self) -> None:
        """
        Log key configuration parameters to MLflow.
        """
        # Log experiment information
        mlflow.log_param("experiment_name", self.experiment_name)

        # Log data configuration
        data_config = self.config_manager.get_data_config()
        if "file" in data_config:
            mlflow.log_param("data.file", data_config["file"])
        if "target_col" in data_config:
            mlflow.log_param("data.target_col", data_config["target_col"])

        # Log model configuration
        model_config = self.config_manager.get_model_config()
        if "type" in model_config:
            mlflow.log_param("model.type", model_config["type"])

        # Log split configuration
        split_config = data_config.get("split", {})
        if "test_size" in split_config:
            mlflow.log_param("split.test_size", split_config["test_size"])
        if "random_state" in split_config:
            mlflow.log_param("split.random_state", split_config["random_state"])

        # Log tags
        for key, value in self.tags.items():
            mlflow.set_tag(key, value)

    def run(self, selected_components: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the full machine learning pipeline with MLflow tracking.

        Args:
            selected_components: Optional list of component names to run.
                                If None, run all enabled components.

        Returns:
            Dictionary with results of the experiment
        """
        start_time = time.time()
        run_id = None
        metrics = {}
        component_results = {}

        # Start MLflow run
        with mlflow.start_run(experiment_id=self.experiment_id, tags=self.tags) as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id}")

            # Log configuration
            self._log_config()

            try:
                # Run each enabled component
                for component_def in self.components:
                    component_name = component_def["name"]
                    component_func = component_def["function"]

                    # Check if component should be run
                    if (selected_components is not None and component_name not in selected_components) or (
                        not self.config_manager.is_component_enabled(component_name)
                    ):
                        logger.info(f"Skipping component: {component_name}")
                        continue

                    logger.info(f"Running component: {component_name}")
                    result = component_func(self.config_manager)
                    component_results[component_name] = result

                    # Check for component error
                    if result.get("status") != "success":
                        error_msg = result.get("error", f"Component {component_name} failed")
                        logger.error(error_msg)
                        if "metrics" in result:
                            metrics.update(result["metrics"])
                        raise RuntimeError(error_msg)

                    # Collect metrics if any
                    if "metrics" in result:
                        metrics.update(result["metrics"])

                # Log total execution time
                total_time = time.time() - start_time
                mlflow.log_metric("total_pipeline_time_seconds", total_time)

                # Successful completion
                logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")

                # Prepare result information
                return {
                    "run_id": run_id,
                    "execution_time": total_time,
                    "status": "success",
                    "metrics": metrics,
                    "component_results": component_results,
                }

            except Exception as e:
                # Log the error
                error_msg = str(e)
                logger.error(f"Pipeline error: {error_msg}")

                # Record error in MLflow
                mlflow.log_param("error", error_msg)
                mlflow.log_param("status", "failed")

                # Log execution time even on failure
                failure_time = time.time() - start_time
                mlflow.log_metric("execution_time_seconds", failure_time)

                # Return error information
                return {
                    "run_id": run_id,
                    "error": error_msg,
                    "execution_time": failure_time,
                    "status": "failed",
                    "metrics": metrics,
                    "component_results": component_results,
                }
