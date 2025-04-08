"""Data manager for ML pipelines.

This module provides a high-level interface for data operations in ML pipelines.
The DataManager class handles data loading, splitting, and basic data operations
while using DataStorage internally for storage operations.
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import DataStorage classes
from src.data.storage import DataStorage, get_storage
from src.models.base_model import BaseModel


class DataManager:
    """
    High-level data management for machine learning workflows.

    Provides a unified interface for data operations including loading,
    splitting, and basic data storage with support for different backends.
    """

    def __init__(self, data_config: Dict[str, Any]):
        """
        Initialize DataManager with configuration.

        Args:
            data_config: Configuration dictionary containing data settings
                Expected keys:
                - storage_type: Storage backend type ('Local', 'GCS', 'BigQuery')
                - storage_config: Additional storage-specific configuration
        """
        self.config = data_config

        # Extract configuration settings
        self.model_dir = data_config.get("model_dir", "models/saved")
        self.results_dir = data_config.get("results_dir", "results")

        # Set up storage
        storage_type = data_config.get("storage_type", "Local")
        storage_config = data_config.get("storage_config", {})

        # Create storage instance
        self.storage = get_storage(
            service=storage_type,
            base_dir=storage_config.get("base_dir", ""),
        )

    def _get_path(self, base_dir: str, file_name: str) -> str:
        """
        Get full path for a file.

        Args:
            base_dir: Base directory
            file_name: File name

        Returns:
            Full path
        """
        return os.path.join(base_dir, file_name)

    def save_data(
        self,
        data: Union[Any, List[Any]],
        file_name: Union[str, List[str]],
        base_dir: str,
        format: str = "pickle",
    ):
        """
        Save one or multiple data items to storage.

        This method provides flexibility to save a single data item or multiple
        data items simultaneously. It supports different storage formats and
        allows specifying a custom base directory.

        Args:
            data: Single data item or list of data items to save
            file_name: Single filename or list of filenames
            format: File format for saving (default is pickle)
            base_dir: Base directory for saving files. If None, uses default processed directory

        Raises:
            ValueError: If the number of data items and filenames do not match
        """
        # Convert single items to lists for uniform processing
        if not isinstance(data, list):
            data = [data]

        if not isinstance(file_name, list):
            file_name = [file_name]

        # Validate that the number of data items matches the number of filenames
        if len(data) != len(file_name):
            raise ValueError("Number of data items must match number of filenames")

        # Ensure the base directory exists
        os.makedirs(base_dir, exist_ok=True)

        # Save each data item
        for item, fname in zip(data, file_name):
            path = self._get_path(base_dir, fname)
            self.storage.save(item, path, format)

    def load_data(
        self,
        base_dir: str,
        file_name: Union[str, List[str]],
        format: str = "pickle",
    ) -> Union[Any, List[Any]]:
        """
        Load one or multiple data items from storage.

        This method provides flexibility to load a single data item or multiple
        data items simultaneously. It supports different storage formats and
        allows specifying a custom base directory.

        Args:
            base_dir: Base directory for loading files
            file_name: Single filename or list of filenames
            format: File format for loading (default is pickle)

        Returns:
            Single data item or list of data items

        Raises:
            FileNotFoundError: If any specified file does not exist
        """
        # Convert single filename to list for uniform processing
        if not isinstance(file_name, list):
            file_name = [file_name]

        # Load data items
        loaded_data = []
        for fname in file_name:
            path = self._get_path(base_dir, fname)

            # Check if file exists
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

            # Load the data
            item = self.storage.load(path, format)
            loaded_data.append(item)

        # Return single item if only one file was loaded, otherwise return list
        return loaded_data[0] if len(loaded_data) == 1 else loaded_data

    def save_model(self, model: BaseModel, file_name: str) -> None:
        """
        Save model using its own save method but with the managed storage path.

        Args:
            model: Model to save (must implement save method)
            file_name: Name to save the model under

        Note:
            This uses the model's own save implementation which handles serialization
        """
        path = self._get_path(self.model_dir, file_name)
        model.save(path)

    def load_model(self, model_class: Type[BaseModel], file_name: str) -> BaseModel:
        """
        Load model using its own class load method but with managed storage path.

        Args:
            model_class: Model class to load (must implement class method load)
            file_name: Name of the model file

        Returns:
            Loaded model instance

        Note:
            This uses the model class's own load implementation which handles deserialization
        """
        path = self._get_path(self.model_dir, file_name)
        return model_class.load(path)

    def save_model_state(self, model_state: Any, file_name: str, format: str = "pickle") -> None:
        """
        Save raw model state data.

        This is an alternative to save_model when you want to handle serialization
        separately from the model's own save method.

        Args:
            model_state: Model state data (already serialized or ready to be pickled)
            file_name: Name of the model file
            format: Format to save in
        """
        path = self._get_path(self.model_dir, file_name)
        self.storage.save(model_state, path, format=format)

    def load_model_state(self, file_name: str, format: str = "pickle") -> Any:
        """
        Load raw model state data.

        This is an alternative to load_model when you want to handle deserialization
        separately from the model's own load method.

        Args:
            file_name: Name of the model file
            format: Format of the file

        Returns:
            Raw model state data
        """
        path = self._get_path(self.model_dir, file_name)
        return self.storage.load(path, format=format)

    def load_results(self, file_name: str, format: str = "csv", **kwargs) -> pd.DataFrame:
        """
        Load results from results directory.

        Args:
            file_name: Name of the results file
            format: Format of the file
            **kwargs: Additional loading parameters

        Returns:
            Loaded results DataFrame
        """
        path = self._get_path(self.results_dir, file_name)
        return self.storage.load(path, format=format, **kwargs)

    def save_results(
        self, results: Union[pd.DataFrame, List, Dict], file_name: str, format: str = "csv", **kwargs
    ) -> None:
        """
        Save results to results directory.

        Args:
            results: Results to save
            file_name: Name of the results file
            format: Format to save in
            **kwargs: Additional saving parameters
        """
        # Convert dict to DataFrame for easier CSV saving if needed
        if isinstance(results, dict) and format == "csv":
            results = pd.DataFrame([results])

        path = self._get_path(self.results_dir, file_name)
        self.storage.save(results, path, format=format, **kwargs)

    def split_data(
        self,
        data: pd.DataFrame,
        target_col: Optional[Union[str, int]] = None,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
        stratify: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.

        Args:
            data: Input DataFrame
            target_col: Target column name or index (defaults to config value)
            test_size: Test set proportion (defaults to config value)
            random_state: Random seed (defaults to config value)
            stratify: Whether to use stratified sampling

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Get parameters from config if not provided
        if target_col is None:
            target_col = self.config.get("target_col", "target")

        if test_size is None:
            test_size = self.config.get("split", {}).get("test_size", 0.2)

        if random_state is None:
            random_state = self.config.get("split", {}).get("random_state", 42)

        # Handle target column
        if isinstance(target_col, int):
            target_col_name = data.columns[target_col]
        else:
            target_col_name = target_col

        # Split data
        X = data.drop(columns=[target_col_name])
        y = data[target_col_name].values

        # Determine if stratification should be used
        stratify_param = y if stratify else None

        # Perform split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )

        return X_train, X_test, y_train, y_test

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute SQL query using BigQuery storage (if available).

        Args:
            query: SQL query string or path to SQL file
            params: Parameters to substitute in the query

        Returns:
            Query results as DataFrame

        Raises:
            ValueError: If storage doesn't support query execution
        """
        if self.config.get("storage_type") != "BigQuery":
            raise ValueError("Query execution is only available with BigQuery storage")

        kwargs = {"params": params} if params else {}
        return self.storage.load(query, **kwargs)

    def get_storage(self) -> DataStorage:
        """
        Get the underlying storage instance.

        Returns:
            The DataStorage instance
        """
        return self.storage
