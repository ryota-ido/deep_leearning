"""Configuration management utility for machine learning pipeline."""

import os
from typing import Any, Dict, Optional

import yaml


class ConfigManager:
    """
    Configuration manager for machine learning pipeline.

    Provides consistent access to configuration parameters with proper defaults
    and validation. Centralizes configuration handling across components.
    """

    def __init__(self, pipeline_config: Dict[str, Any], experiment_config: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration manager.

        Args:
            pipeline_config: Pipeline configuration dictionary
            experiment_config: Experiment configuration dictionary (optional)
        """
        self.pipeline_config = pipeline_config or {}
        self.experiment_config = experiment_config or {}

        # Merged configuration for unified access
        self.config = self._merge_configs()

        # Standard output paths with proper defaults
        self.output_paths = self._setup_output_paths()

    def _merge_configs(self) -> Dict[str, Any]:
        """
        Merge pipeline and experiment configurations with proper precedence.

        Returns:
            Merged configuration dictionary
        """
        merged = {
            "pipeline": self.pipeline_config,
            "experiment": self.experiment_config,
        }

        # Extract MLflow config (prioritize pipeline config over experiment)
        mlflow_config = {}
        if "mlflow" in self.experiment_config:
            mlflow_config.update(self.experiment_config["mlflow"])
        if "mlflow" in self.pipeline_config:
            mlflow_config.update(self.pipeline_config["mlflow"])

        merged["mlflow"] = mlflow_config

        return merged

    def _setup_output_paths(self) -> Dict[str, str]:
        """
        Set up standardized output paths with proper defaults.

        Returns:
            Dictionary of output paths
        """
        base_output_dir = self.pipeline_config.get("output", {}).get("base_dir", "out")

        # Define all output paths with consistent defaults
        paths = {
            "data": {
                "raw": os.path.join(base_output_dir, "data/raw"),
                "split": os.path.join(base_output_dir, "data/split"),
                "processed": os.path.join(base_output_dir, "data/processed"),
            },
            "models": {
                "saved": os.path.join(base_output_dir, "models/saved"),
                "tuned": os.path.join(base_output_dir, "models/tuned"),
            },
            "tuning": {
                "best_params": os.path.join(base_output_dir, "tuning/best_params"),
                "history": os.path.join(base_output_dir, "tuning/history"),
            },
            "logs": os.path.join(base_output_dir, "logs"),
            "results": os.path.join(base_output_dir, "results"),
        }

        # Override with user-specified paths if present
        output_config = self.pipeline_config.get("output", {})

        if "data" in output_config and isinstance(output_config["data"], dict):
            for key, value in output_config["data"].items():
                if key in paths["data"]:
                    paths["data"][key] = os.path.join(base_output_dir, value)

        if "models" in output_config and isinstance(output_config["models"], dict):
            for key, value in output_config["models"].items():
                if key in paths["models"]:
                    paths["models"][key] = os.path.join(base_output_dir, value)

        if "tuning" in output_config and isinstance(output_config["tuning"], dict):
            for key, value in output_config["tuning"].items():
                if key in paths["tuning"]:
                    paths["tuning"][key] = os.path.join(base_output_dir, value)

        if "logs" in output_config:
            paths["logs"] = output_config["logs"]

        if "results" in output_config:
            paths["results"] = output_config["results"]

        return paths

    def get(self, section: str, key: str = None, default: Any = None) -> Any:
        """
        Get a configuration value from the merged configuration.

        Args:
            section: Configuration section name
            key: Optional key within the section. If None, returns the entire section.
            default: Default value if the section or key is not found

        Returns:
            Configuration value or default
        """
        if section not in self.config:
            return default

        section_data = self.config[section]

        if key is None:
            return section_data

        if isinstance(section_data, dict) and key in section_data:
            return section_data[key]

        return default

    def get_output_path(self, category: str, subcategory: Optional[str] = None) -> str:
        """
        Get standardized output path.

        Args:
            category: Path category ('data', 'models', 'tuning', 'logs', 'results')
            subcategory: Optional subcategory ('raw', 'processed', 'saved', etc.)

        Returns:
            Output path string

        Raises:
            ValueError: If category or subcategory is invalid
        """
        if category not in self.output_paths:
            raise ValueError(f"Invalid path category: {category}")

        if subcategory is None:
            if isinstance(self.output_paths[category], dict):
                raise ValueError(f"Subcategory required for {category}")
            return self.output_paths[category]

        if not isinstance(self.output_paths[category], dict):
            raise ValueError(f"Category {category} does not support subcategories")

        if subcategory not in self.output_paths[category]:
            raise ValueError(f"Invalid subcategory {subcategory} for {category}")

        return self.output_paths[category][subcategory]

    def get_experiment_param(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get parameter from experiment configuration.

        Args:
            section: Configuration section
            key: Parameter key
            default: Default value if parameter is not found

        Returns:
            Parameter value or default
        """
        return self.experiment_config.get(section, {}).get(key, default)

    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration section."""
        return self.experiment_config.get("data", {})

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration section."""
        return self.experiment_config.get("preprocessing", {})

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration section."""
        return self.experiment_config.get("model", {})

    def get_tuning_config(self) -> Dict[str, Any]:
        """Get tuning configuration section."""
        return self.experiment_config.get("tuning", {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration section."""
        return self.experiment_config.get("evaluation", {})

    def get_cross_validation_config(self) -> Dict[str, Any]:
        """Get cross-validation configuration section."""
        return self.experiment_config.get("cross_validation", {})

    def get_mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow configuration (merged from both configs)."""
        return self.config.get("mlflow", {})

    def is_component_enabled(self, component_name: str) -> bool:
        """
        Check if a pipeline component is enabled.

        Args:
            component_name: Name of the component to check

        Returns:
            True if the component is enabled, False otherwise
        """
        components = self.pipeline_config.get("pipeline", {}).get("components", [])
        for component in components:
            if component.get("name") == component_name:
                return component.get("enabled", True)
        return True  # Default to enabled if not specified

    @classmethod
    def from_yaml_files(
        cls, pipeline_config_path: str, experiment_config_path: Optional[str] = None
    ) -> "ConfigManager":
        """
        Create ConfigManager from YAML files.

        Args:
            pipeline_config_path: Path to pipeline configuration YAML
            experiment_config_path: Path to experiment configuration YAML (optional)

        Returns:
            ConfigManager instance

        Raises:
            FileNotFoundError: If configuration files are not found
            ValueError: If YAML parsing fails
        """
        pipeline_config = {}
        experiment_config = {}

        # Load pipeline config
        try:
            with open(pipeline_config_path, "r") as f:
                pipeline_config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Pipeline configuration file not found: {pipeline_config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing pipeline configuration: {e}")

        if not experiment_config_path and pipeline_config.get("experiment_config_path"):
            experiment_config_path = pipeline_config.get("experiment_config_path")

        # Load experiment config if provided
        if experiment_config_path:
            try:
                with open(experiment_config_path, "r") as f:
                    experiment_config = yaml.safe_load(f) or {}
            except FileNotFoundError:
                raise FileNotFoundError(f"Experiment configuration file not found: {experiment_config_path}")
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing experiment configuration: {e}")

        return cls(pipeline_config, experiment_config)
