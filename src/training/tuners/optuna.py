"""Grid search hyperparameter tuning module."""

import time
from typing import Any, Dict, Optional

import optuna
from optuna import Trial

from src.training.cross_validation.cross_validation import CrossValidator
from src.training.trainer import Trainer
from src.training.tuners.base_tuner import BaseTuner
from src.utils.registry import register_tuner


@register_tuner("optuna")
class OptunaTuner(BaseTuner):
    """
    Optuna-based hyperparameter tuning implementation.

    Uses Optuna to perform efficient hyperparameter search via
    Bayesian optimization or other sampling methods.
    """

    def __init__(self, trainer: Trainer, tuning_config: Optional[Dict[str, Any]] = None):
        """
        Initialize Optuna tuner.

        Args:
            trainer: Trainer instance to be tuned
            tuning_config: Configuration dictionary for Optuna parameters
        """
        super().__init__(trainer, tuning_config)

        # self.params = self._convert_param_space(self.config.get("params", {}))
        self.params = self.config.get("params", {})
        self.n_trials = self.config.get("n_trials", 50)
        self.timeout = self.config.get("timeout", None)
        self.direction = self.config.get("direction", "maximize")
        self.scoring_metric = self.config.get("scoring_metric", None)

        if not self.cross_validator:
            self.cross_validator = CrossValidator(trainer)

        if self.scoring_metric:
            self.cross_validator.scoring_metric = self.scoring_metric

        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate Optuna-specific configuration.
        """
        super()._validate_config()

        if not self.params:
            raise ValueError("Parameter space must be specified for Optuna tuning.")

    def _create_trial_params(self, trial: Trial) -> Dict[str, Any]:
        """
        Create parameters for a specific trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of parameter values for the trial
        """
        params = {}
        for param_name, config in self.params.items():
            method_name = config[0]
            args = config[1]
            kwargs = {}

            # Extract kwargs if the last element is a dictionary
            if len(args) > 2 and isinstance(args[-1], dict):
                kwargs = args[-1]
                args = args[:-1]

            # Get the suggestion method from the Trial object
            suggest_method = getattr(trial, method_name)

            # Call the method with the parameter name, arguments and keyword arguments
            params[param_name] = suggest_method(param_name, *args, **kwargs)

        return params

    def _objective(self, trial: Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Evaluation score for the trial
        """
        # Apply parameters to the model
        params = self._create_trial_params(trial)
        self.trainer.set_params(**params)

        # Perform cross-validation with these parameters
        cv_results = self.cross_validator.cross_validate(self.X, self.y)

        # Return the mean score as optimization target
        return cv_results["mean_score"]

    def tune(self, X, y):
        """
        Run Optuna hyperparameter optimization.

        Args:
            X: Training features
            y: Target values

        Returns:
            Dictionary with best hyperparameters and score
        """
        # Store data for use in objective function
        self.X = X
        self.y = y

        start_time = time.time()

        # Create and run the study
        study = optuna.create_study(direction=self.direction)

        try:
            study.optimize(self._objective, n_trials=self.n_trials, timeout=self.timeout)

            self.best_params = study.best_params
            self.best_score = study.best_value

            # Store comprehensive results
            self.results = {
                "best_params": self.best_params,
                "best_score": self.best_score,
                "study": study,
                "duration": time.time() - start_time,
            }

            # Apply best parameters to the model
            self.trainer.set_params(**self.best_params)

            return {"best_params": self.best_params, "best_score": self.best_score}

        except Exception as e:
            self.results = {
                "error": str(e),
                "duration": time.time() - start_time,
            }
            raise
