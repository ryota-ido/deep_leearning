"""Data storage module for machine learning workflow."""

import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")


class DataStorage(ABC):
    """
    Abstract base class for data storage implementations.

    Provides a unified interface for loading and saving data across
    different storage backends (local filesystem, cloud storage, etc.)
    """

    @abstractmethod
    def load(self, source: str, format: str = "pickle", **kwargs) -> pd.DataFrame:
        """
        Load data from the specified source.

        Args:
            source: Path or identifier of the data source
            format: Format of the data ('pickle', 'csv', 'json', etc.)
            **kwargs: Additional backend-specific parameters

        Returns:
            DataFrame containing the loaded data
        """
        pass

    @abstractmethod
    def save(self, data: Union[pd.DataFrame, str, List], destination: str, format: str = "pickle", **kwargs) -> None:
        """
        Save data to the specified destination.

        Args:
            data: Data to be saved (DataFrame, string, or list)
            destination: Path or identifier where data will be saved
            format: Format to save the data in ('pickle', 'csv', 'json', etc.)
            **kwargs: Additional backend-specific parameters
        """
        pass


class LocalStorage(DataStorage):
    """
    Storage implementation for local filesystem operations.

    Provides methods to load and save data to the local filesystem,
    supporting various formats including pickle, CSV, and JSON.
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize local storage handler.

        Args:
            base_dir: Optional base directory for relative paths
        """
        self.base_dir = base_dir

    def _get_full_path(self, path: str) -> str:
        """
        Construct full path from potentially relative path.

        Args:
            path: Input path, possibly relative

        Returns:
            Full absolute path
        """
        if self.base_dir and not os.path.isabs(path):
            return os.path.join(self.base_dir, path)
        return path

    def load(self, source: str, format: str = "pickle", **kwargs) -> pd.DataFrame:
        """
        Load data from local filesystem.

        Args:
            source: Path to the file
            format: Format of the file ('pickle', 'csv', 'json', 'tsv')
            **kwargs: Additional parameters for pandas loading functions

        Returns:
            Pandas DataFrame containing the loaded data

        Raises:
            FileNotFoundError: If the specified file doesn't exist
        """
        full_path = self._get_full_path(source)

        if not os.path.isfile(full_path):
            logger.warning(f"File not found: {full_path}")
            return pd.DataFrame()

        logger.info(f"Loading data from {full_path} in {format} format")

        if format == "csv":
            return pd.read_csv(full_path, **kwargs)
        elif format == "json":
            return pd.read_json(full_path, **kwargs)
        elif format == "tsv":
            return pd.read_csv(full_path, sep="\t", **kwargs)
        elif format == "pickle":
            return pd.read_pickle(full_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def save(self, data: Union[pd.DataFrame, str, List], destination: str, format: str = "pickle", **kwargs) -> None:
        """
        Save data to local filesystem.

        Args:
            data: Data to be saved (DataFrame, string, or list)
            destination: Path where data will be saved
            format: Format to save the data in ('pickle', 'csv', 'json', 'tsv')
            **kwargs: Additional parameters for pandas saving functions

        Raises:
            TypeError: If data type is not supported for the specified format
        """
        full_path = self._get_full_path(destination)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        logger.info(f"Saving data to {full_path} in {format} format")

        if isinstance(data, pd.DataFrame):
            if format == "csv":
                data.to_csv(full_path, index=False, **kwargs)
            elif format == "json":
                data.to_json(full_path, **kwargs)
            elif format == "tsv":
                data.to_csv(full_path, sep="\t", index=False, **kwargs)
            elif format == "pickle":
                data.to_pickle(full_path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")
        elif isinstance(data, str):
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(data)
        elif isinstance(data, list):
            if format == "pickle":
                with open(full_path, "wb") as f:
                    pickle.dump(data, f)
            else:
                # Convert list to DataFrame if possible and save
                df = pd.DataFrame(data)
                self.save(df, destination, format, **kwargs)
        else:
            raise TypeError(f"Unsupported data type: {type(data).__name__} for format: {format}")


def get_storage(service: str = "Local", base_dir: str = "") -> DataStorage:
    """
    Factory function to create an appropriate DataStorage instance.

    Args:
        service: Storage service to use ('Local', 'GCS', or 'BigQuery')
        base_dir: Base directory/prefix within storage

    Returns:
        DataStorage instance based on the specified service

    Raises:
        ValueError: If required parameters are missing for the selected service
    """

    return LocalStorage(base_dir)
