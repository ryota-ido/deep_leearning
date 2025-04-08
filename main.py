"""Main entry point for the machine learning framework."""

import argparse

from src.experiment import run_experiment

# Model-related imports
from src.models.base_model import BaseModel

# Preprocess components
from src.preprocess.base_preprocessor import BasePreprocessor

# Training components
from src.training.tuners.base_tuner import BaseTuner

# Utility imports
from src.utils.logger import setup_logger
from src.utils.registry import model_registry, preprocessor_registry, tuner_registry


def main():
    """
    Main function to parse arguments and run machine learning experiments.

    Handles command-line argument parsing, configuration loading,
    and experiment execution.
    """
    # Configure logging
    logger = setup_logger("ml_framework_main", level="INFO")

    try:
        # Import modules to ensure all classes are registered
        logger.info("Discovering and registering models and tuners...")
        model_registry.discover_modules(base_package="src.models", base_class=BaseModel)
        tuner_registry.discover_modules(base_package="src.training.tuners", base_class=BaseTuner)
        preprocessor_registry.discover_modules(base_package="src.preprocess", base_class=BasePreprocessor)

        # Set up argument parser
        parser = argparse.ArgumentParser(description="Machine Learning Framework Experiment Runner")
        parser.add_argument(
            "--config",
            type=str,
            default="config/config.yaml",
            help="Path to the configuration file for the experiment",
        )
        parser.add_argument("--model-config", type=str, help="Path to a specific model configuration file")

        # Parse arguments
        args = parser.parse_args()

        # Load main configuration
        logger.info(f"Loading configuration from {args.config}")

        # Run experiment
        logger.info("Starting machine learning experiment...")
        run_experiment(args.config)

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
