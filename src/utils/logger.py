"""Logging utilities for machine learning experiments."""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up and configure a logger with flexible options.

    Args:
        name: Name of the logger
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional path to log file for file-based logging
        console_output: Whether to output logs to the console

    Returns:
        Configured logging.Logger instance
    """
    # Map string levels to logging module levels
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    # Normalize and validate log level
    log_level = level_map.get(level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Remove existing handlers to prevent duplicate logs
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log file is specified
    if log_file:
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MLLogger:
    """
    Specialized logger for machine learning experiments.

    Provides comprehensive logging capabilities tailored to
    machine learning workflow tracking.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ML experiment logger.

        Args:
            config: Configuration dictionary containing logging parameters
        """
        self.config = config

        # Extract logging parameters
        log_level = config.get("logging", {}).get("level", "INFO")
        log_dir = config.get("logging", {}).get("save_dir", "logs")
        experiment_name = config.get("experiment", {}).get("name", "default_experiment")

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Set up logger
        log_file = os.path.join(log_dir, f"{experiment_name}.log")
        self.logger = setup_logger(f"ml_experiment_{experiment_name}", level=log_level, log_file=log_file)

    def log_config(self) -> None:
        """
        Log the entire experiment configuration.

        Provides a detailed overview of all configuration parameters.
        """
        self.logger.info("Starting experiment with configuration:")

        # Log model config path if present
        if "model_config" in self.config:
            self.logger.info(f"  Using model configuration from: {self.config['model_config']}")

        # Log all configuration sections
        for section, params in self.config.items():
            if section == "model_config":
                continue  # Already logged above

            self.logger.info(f"  {section}:")
            if isinstance(params, dict):
                self._log_dict_items(params, indent=2)
            else:
                self.logger.info(f"    {params}")

    def _log_dict_items(self, items: Dict[str, Any], indent: int = 0) -> None:
        """
        Recursively log dictionary items with proper indentation.

        Args:
            items: Dictionary to log
            indent: Current indentation level
        """
        indent_str = "  " * indent
        for key, value in items.items():
            if isinstance(value, dict):
                self.logger.info(f"{indent_str}{key}:")
                self._log_dict_items(value, indent + 1)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                # Handle lists of dictionaries
                self.logger.info(f"{indent_str}{key}: [")
                for i, item in enumerate(value):
                    self.logger.info(f"{indent_str}  Item {i}:")
                    self._log_dict_items(item, indent + 2)
                self.logger.info(f"{indent_str}]")
            else:
                self.logger.info(f"{indent_str}{key}: {value}")

    def log_data_info(
        self, data_shape: Tuple[int, int], feature_names: List[Union[str, int]], target_name: Union[str, int]
    ) -> None:
        """
        Log detailed information about the loaded dataset.

        Args:
            data_shape: Shape of the dataset (rows, columns)
            feature_names: List of feature column names
            target_name: Name of the target variable
        """
        self.logger.info(f"Data loaded with shape: {data_shape}")
        self.logger.info(f"Target variable: {target_name}")
        self.logger.info(f"Number of features: {len(feature_names)}")

        if len(feature_names) <= 20:
            self.logger.info(f"Feature names: {', '.join([str(f) for f in feature_names])}")

    def log_preprocessing(self, steps: List[str]) -> None:
        """
        Log preprocessing steps applied to the data.

        Args:
            steps: List of preprocessing transformations
        """
        self.logger.info("Applied preprocessing steps:")
        for i, step in enumerate(steps, 1):
            self.logger.info(f"  {i}. {step}")

    def log_model_info(self, model_name: str, params: Dict[str, Any]) -> None:
        """
        Log information about the initialized model.

        Args:
            model_name: Name of the machine learning model
            params: Dictionary of model parameters
        """
        self.logger.info(f"Model: {model_name}")
        self.logger.info("Model parameters:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")

    def log_training_start(self) -> None:
        """Log the beginning of model training process."""
        self.logger.info("Starting model training...")

    def log_training_end(self, duration: float) -> None:
        """
        Log the completion of model training.

        Args:
            duration: Total training time in seconds
        """
        self.logger.info(f"Model training completed in {duration:.2f} seconds")

    def log_metrics(self, metrics: Dict[str, float], phase: str = "evaluation") -> None:
        """
        Log performance metrics for the model.

        Args:
            metrics: Dictionary of metric names and their values
            phase: Phase of evaluation (e.g., "training", "validation", "testing")
        """
        self.logger.info(f"{phase.capitalize()} metrics:")
        for name, value in metrics.items():
            self.logger.info(f"  {name}: {value:.4f}")

    def log_hyperparameter_tuning(self, best_params: Dict[str, Any], best_score: float) -> None:
        """
        Log results of hyperparameter tuning.

        Args:
            best_params: Dictionary of best hyperparameters found
            best_score: Best performance score achieved
        """
        self.logger.info("Hyperparameter tuning completed")
        self.logger.info(f"Best score: {best_score:.4f}")
        self.logger.info("Best parameters:")
        for name, value in best_params.items():
            self.logger.info(f"  {name}: {value}")

    def log_error(self, error: Exception) -> None:
        """
        Log an error with full traceback.

        Args:
            error: Exception to be logged
        """
        self.logger.error(f"Error: {str(error)}", exc_info=True)

    def log_warning(self, message: str) -> None:
        """
        Log a warning message.

        Args:
            message: Warning message to log
        """
        self.logger.warning(message)

    def log_message(self, message: str, level: str = "info") -> None:
        """
        Log a custom message with specified logging level.

        Args:
            message: Message to log
            level: Logging level ('debug', 'info', 'warning', 'error', 'critical')
        """
        level = level.lower()
        log_method = getattr(self.logger, level, self.logger.info)
        log_method(message)
