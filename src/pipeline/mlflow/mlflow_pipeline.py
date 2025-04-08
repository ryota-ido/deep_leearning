"""MLflow pipeline for machine learning experiments with component isolation."""

import argparse
import logging
import sys
from typing import Any, Dict, List, Optional

from src.pipeline.mlflow.core import MLflowPipeline
from src.utils.config_manager import ConfigManager

# Configure logging
logger = logging.getLogger(__name__)


def run_pipeline(
    config_path: str, experiment_config_path: Optional[str] = None, selected_components: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run the MLflow pipeline with configuration from file(s).

    Args:
        config_path: Path to pipeline configuration file (YAML format)
        experiment_config_path: Path to experiment configuration file (optional)
        selected_components: Optional list of component names to run

    Returns:
        Dictionary with results from the pipeline execution
    """
    try:
        # Load configuration using ConfigManager
        logger.info(f"Loading configuration from {config_path}")
        if experiment_config_path:
            logger.info(f"Loading experiment configuration from {experiment_config_path}")

        config_manager = ConfigManager.from_yaml_files(config_path, experiment_config_path)

        # Create and run pipeline
        logger.info("Initializing MLflow pipeline")
        pipeline = MLflowPipeline(config_manager)

        logger.info("Starting pipeline execution")
        if selected_components:
            logger.info(f"Running selected components: {', '.join(selected_components)}")
            results = pipeline.run(selected_components)
        else:
            results = pipeline.run()

        # Check results
        if results.get("status") == "success":
            logger.info(f"Pipeline completed successfully. Run ID: {results.get('run_id')}")
        else:
            logger.error(f"Pipeline failed: {results.get('error')}")

        return results

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"Invalid configuration: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in pipeline execution: {str(e)}")
        raise


if __name__ == "__main__":
    # Configure logging for script execution
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run MLflow pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to pipeline configuration file")
    parser.add_argument("--experiment-config", type=str, help="Optional override for experiment configuration path")
    parser.add_argument(
        "--components", type=str, help="Comma-separated list of components to run (e.g., data_loader,training)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Set logging level from args
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Parse selected components if provided
    selected_components = None
    if args.components:
        selected_components = [comp.strip() for comp in args.components.split(",")]

    # Run the pipeline
    try:
        results = run_pipeline(args.config, args.experiment_config, selected_components)

        # Display results
        if results.get("status") == "success":
            print(f"\nPipeline completed successfully.")
            print(f"MLflow Run ID: {results['run_id']}")
            print("\nMetrics:")
            for name, value in results.get("metrics", {}).items():
                if isinstance(value, float):
                    print(f"  {name}: {value:.4f}")
                else:
                    print(f"  {name}: {value}")
            print(f"\nExecution time: {results['execution_time']:.2f} seconds")
        else:
            print(f"\nPipeline failed: {results.get('error')}")
            print(f"MLflow Run ID: {results.get('run_id', 'N/A')}")

    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        sys.exit(1)
