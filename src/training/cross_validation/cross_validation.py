"""Cross-validation module for model evaluation."""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from src.training.trainer import Trainer


class CrossValidator:
    """
    Cross-validation implementation for robust model evaluation.

    Supports multiple splitting strategies including K-Fold,
    Stratified K-Fold, and Time Series cross-validation.
    """

    def __init__(self, trainer: Trainer, config: Optional[Dict[str, Any]] = None):
        """
        Initialize cross-validator with a trainer and configuration.

        Args:
            trainer: Trainer instance to be evaluated
            config: Configuration dictionary for cross-validation parameters
                    Allows customization of cross-validation strategy
        """
        default_config: Dict[str, Any] = {
            "split_method": "kfold",  # Default splitting method
            "n_splits": 5,  # Number of folds
            "shuffle": True,  # Whether to shuffle data before splitting
            "random_state": 42,  # Random seed for reproducibility
            "scoring_metric": "accuracy_score",  # Default scoring metric
        }
        self.trainer = trainer
        self.config = {**default_config, **(config or {})}
        self.scoring_metric = self.config["scoring_metric"]

        # Configure cross-validation splitter based on specified method
        self.splitter = self._get_splitter()

        # Store results from each fold
        self.fold_results: List[float] = []
        self.fold_models: List[Any] = []

    def _get_splitter(self):
        """
        Select and configure the appropriate cross-validation splitter.

        Returns:
            Configured cross-validation splitter based on configuration

        Raises:
            ValueError: If an unsupported split method is specified
        """
        split_method = self.config["split_method"].lower()

        if split_method == "kfold":
            return KFold(
                n_splits=self.config["n_splits"],
                shuffle=self.config["shuffle"],
                random_state=self.config["random_state"],
            )
        elif split_method == "stratifiedkfold":
            return StratifiedKFold(
                n_splits=self.config["n_splits"],
                shuffle=self.config["shuffle"],
                random_state=self.config["random_state"],
            )
        elif split_method == "timeseries":
            return TimeSeriesSplit(n_splits=self.config["n_splits"])
        else:
            raise ValueError(f"Unsupported cross-validation method: {split_method}")

    def cross_validate(self, X: np.ndarray, y: Optional[np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Perform cross-validation on the given dataset.

        Splits the data into training and validation sets, trains the model
        on each split, and evaluates its performance.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            Dictionary containing cross-validation results, including:
            - fold_scores: Scores for each fold
            - mean_score: Average score across all folds
            - best_model: Model with median performance
        """
        splits = list(self.splitter.split(X, y))

        # Reset fold results before starting
        self.fold_results = []
        self.fold_models = []

        # Perform cross-validation
        for train_index, val_index in splits:
            # Split data into training and validation sets
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Train model on each fold
            self.trainer.train(X_train, y_train, reset=True, **kwargs)

            # Evaluate and store results
            score = self._score(self.trainer, X_val, y_val)
            model = self.trainer.model

            # Store fold results
            self.fold_results.append(score)
            self.fold_models.append(model)

        # Select best model based on median score
        median_index = np.argsort(self.fold_results)[len(self.fold_results) // 2]
        best_model = self.fold_models[median_index]

        # Update trainer with best model
        self.trainer.model = best_model

        return {"fold_scores": self.fold_results, "mean_score": np.mean(self.fold_results), "best_model": best_model}

    def _score(self, trainer: Trainer, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the score for a specific fold using the specified metric.

        Args:
            trainer: Trainer instance with trained model
            X: Validation feature matrix
            y: Validation target values

        Returns:
            Score value based on the configured scoring metric
        """
        # Get evaluation results and extract the specified metric
        evaluation_results = trainer.evaluate(X, y)
        return evaluation_results.get(
            self.scoring_metric,
            list(evaluation_results.values())[0],  # Default to first metric if specified not found
        )
