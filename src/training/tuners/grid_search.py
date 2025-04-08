"""Grid search hyperparameter tuning module."""

import time
from typing import Any, Dict, Optional

from sklearn.model_selection import GridSearchCV

from src.training.trainer import Trainer
from src.training.tuners.base_tuner import BaseTuner
from src.utils.registry import register_tuner


@register_tuner("grid_search")
class GridSearchTuner(BaseTuner):
    """
    Grid search hyperparameter tuning implementation.

    Performs exhaustive search over specified parameter values
    to find the best combination of hyperparameters.
    """

    def __init__(self, trainer: Trainer, tuning_config: Optional[Dict[str, Any]] = None):
        """
        Initialize grid search tuner.

        Args:
            trainer: Trainer instance to be tuned
            config: Configuration dictionary for grid search parameters
        """
        super().__init__(trainer, tuning_config)

        # Extract grid search specific configuration
        self.params = self.config.get("params", {})
        if self.cross_validator:
            self.cv = self.cross_validator.splitter or 5
        else:
            self.cv = 5

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate grid search specific configuration.

        Checks that necessary parameters for grid search are present.

        Raises:
            ValueError: If required configuration is missing or invalid
        """
        super()._validate_config()

        if not self.params:
            raise ValueError("Parameter grid must be specified for grid search.")

        if not isinstance(self.params, dict):
            raise ValueError("Parameter grid must be a dictionary.")

    def tune(self, X, y) -> Dict[str, Any]:
        """
        Perform grid search hyperparameter tuning.

        Exhaustively searches through all parameter combinations
        to find the optimal set of hyperparameters.

        Args:
            X: Training feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            Dictionary containing:
            - best_params: Optimal hyperparameters
            - best_score: Best performance score
            - cv_results: Detailed cross-validation results
        """
        # Start timing the tuning process
        start_time = time.time()

        # Create a copy of the original model for grid search
        estimator = self.trainer.model._model

        try:
            # Set up and perform grid search
            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=self.params,
                cv=self.cv,
                scoring=self.scoring_metric,
            )

            # Fit grid search
            grid_search.fit(X, y)

            # Store tuning results
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_

            # Comprehensive results dictionary
            self.results = {
                "best_params": self.best_params,
                "best_score": self.best_score,
                "cv_results": grid_search.cv_results_,
                "duration": time.time() - start_time,
            }

            return {"best_params": self.best_params, "best_score": self.best_score}

        except Exception as e:
            # Log and re-raise any errors during tuning
            self.results = {
                "error": str(e),
                "duration": time.time() - start_time,
            }
            raise
