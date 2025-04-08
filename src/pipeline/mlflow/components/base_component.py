"""Base component for machine learning pipeline."""

import logging
import time

import mlflow

from src.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


def component(component_name: str):
    """
    Decorator for functional pipeline components.

    Provides the same functionality as BaseComponent for function-based components.

    Args:
        component_name: Name of the component

    Returns:
        Decorator function
    """

    def decorator(func):
        def wrapper(config_manager: ConfigManager, *args, **kwargs):
            start_time = time.time()

            try:
                # Log component start
                logger.info(f"Starting component: {component_name}")

                # Execute component function
                result = func(config_manager, *args, **kwargs)

                # Calculate execution time
                execution_time = time.time() - start_time

                # Log execution time
                mlflow.log_metric(f"{component_name}_time_seconds", execution_time)
                logger.info(f"Component {component_name} completed in {execution_time:.2f} seconds")

                # Add status to result if not already present
                if isinstance(result, dict) and "status" not in result:
                    result["status"] = "success"

                # Return the result
                return result

            except Exception as e:
                # Log error
                error_msg = str(e)
                logger.error(f"Error in component {component_name}: {error_msg}")
                mlflow.log_param(f"{component_name}_error", error_msg)

                # Log execution time even on failure
                execution_time = time.time() - start_time
                mlflow.log_metric(f"{component_name}_time_seconds", execution_time)

                # Return error information
                return {"status": "error", "error": error_msg, "execution_time": execution_time}

        return wrapper

    return decorator
