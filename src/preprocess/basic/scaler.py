"""Data scaling implementations for preprocessing."""

from typing import Any, Dict, List, Optional

from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import RobustScaler as SklearnRobustScaler
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from src.preprocess.base_preprocessor import BasePreprocessor
from src.utils.registry import register_preprocessor


class BaseScaler(BasePreprocessor):
    """
    Base class for all scalers in the preprocessing framework.

    Implements common functionality for different scaling approaches
    while delegating specific scaling logic to subclasses.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, columns: Optional[List[str]] = None):
        """
        Initialize the base scaler.

        Args:
            config: Configuration dictionary
            columns: Specific columns to scale (None = all numeric columns)
        """
        super().__init__(config)

        # Configuration
        self.columns = config.get("columns", columns) if config else columns

        # Scalers for each column
        self.scalers: Dict[str, Any] = {}

    def fit(self, data, target=None):
        """
        Learn scaling parameters from training data.

        Args:
            data: Input DataFrame
            target: Optional target variable (not used for scaling)

        Returns:
            Self reference for method chaining
        """
        # Determine which columns to process
        if self.columns is None:
            # Get all numeric columns
            columns_to_process = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        else:
            # Use specified columns that exist in the data
            columns_to_process = [col for col in self.columns if col in data.columns]

        for col in columns_to_process:
            # Skip columns with too many NaN values
            if data[col].isna().sum() > len(data) * 0.5:
                continue

            # Create scaler for this column
            scaler = self._create_scaler()

            # Handle potential NaN values
            valid_data = data[col].dropna().values.reshape(-1, 1)

            if len(valid_data) > 0:
                scaler.fit(valid_data)
                self.scalers[col] = scaler

        self.fitted = True
        return self

    def transform(self, data):
        """
        Apply scaling to numerical columns based on learned parameters.

        Args:
            data: Input DataFrame to scale

        Returns:
            DataFrame with scaled values

        Raises:
            ValueError: If the scaler hasn't been fitted yet
        """
        self._validate_fitted()

        # Create a copy to avoid modifying the original data
        result = data.copy()

        # Apply scaling to each column
        for col, scaler in self.scalers.items():
            if col in result.columns:
                # Handle NaN values - scale only non-NaN values
                mask = result[col].notna()
                if mask.any():
                    result.loc[mask, col] = scaler.transform(result.loc[mask, col].values.reshape(-1, 1)).flatten()

        return result

    def reset(self):
        """
        Reset fitted state and learned parameters.

        Returns:
            Self reference for method chaining
        """
        self.scalers = {}
        self.fitted = False
        return self

    def _create_scaler(self) -> Any:
        """
        Create the specific scaler instance.

        This method should be implemented by subclasses.

        Returns:
            Scaler instance
        """
        raise NotImplementedError("Subclasses must implement _create_scaler method")


@register_preprocessor("standard_scaler")
class StandardScaler(BaseScaler):
    """
    Standardizes features by removing the mean and scaling to unit variance.

    Uses the StandardScaler from scikit-learn, applying it to specified columns.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        with_mean: bool = True,
        with_std: bool = True,
    ):
        """
        Initialize StandardScaler.

        Args:
            config: Configuration dictionary
            columns: Specific columns to scale (None = all numeric columns)
            with_mean: Whether to center the data before scaling
            with_std: Whether to scale the data to unit variance
        """
        super().__init__(config, columns)

        # StandardScaler specific parameters
        if config:
            self.with_mean = config.get("with_mean", with_mean)
            self.with_std = config.get("with_std", with_std)
        else:
            self.with_mean = with_mean
            self.with_std = with_std

    def _create_scaler(self) -> SklearnStandardScaler:
        """
        Create a StandardScaler instance with configured parameters.

        Returns:
            StandardScaler instance
        """
        return SklearnStandardScaler(with_mean=self.with_mean, with_std=self.with_std)


@register_preprocessor("min_max_scaler")
class MinMaxScaler(BaseScaler):
    """
    Scales features to a specified range, typically [0, 1].

    Uses the MinMaxScaler from scikit-learn, applying it to specified columns.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        feature_range: tuple = (0, 1),
    ):
        """
        Initialize MinMaxScaler.

        Args:
            config: Configuration dictionary
            columns: Specific columns to scale (None = all numeric columns)
            feature_range: Desired range of transformed data
        """
        super().__init__(config, columns)

        # MinMaxScaler specific parameters
        if config:
            self.feature_range = config.get("feature_range", feature_range)
        else:
            self.feature_range = feature_range

    def _create_scaler(self) -> SklearnMinMaxScaler:
        """
        Create a MinMaxScaler instance with configured parameters.

        Returns:
            MinMaxScaler instance
        """
        return SklearnMinMaxScaler(feature_range=self.feature_range)


@register_preprocessor("robust_scaler")
class RobustScaler(BaseScaler):
    """
    Scales features using statistics that are robust to outliers.

    Uses the RobustScaler from scikit-learn, applying it to specified columns.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: tuple = (25.0, 75.0),
    ):
        """
        Initialize RobustScaler.

        Args:
            config: Configuration dictionary
            columns: Specific columns to scale (None = all numeric columns)
            with_centering: Whether to center the data before scaling
            with_scaling: Whether to scale the data to unit variance
            quantile_range: Quantile range used to calculate scale
        """
        super().__init__(config, columns)

        # RobustScaler specific parameters
        if config:
            self.with_centering = config.get("with_centering", with_centering)
            self.with_scaling = config.get("with_scaling", with_scaling)
            self.quantile_range = config.get("quantile_range", quantile_range)
        else:
            self.with_centering = with_centering
            self.with_scaling = with_scaling
            self.quantile_range = quantile_range

    def _create_scaler(self) -> SklearnRobustScaler:
        """
        Create a RobustScaler instance with configured parameters.

        Returns:
            RobustScaler instance
        """
        return SklearnRobustScaler(
            with_centering=self.with_centering, with_scaling=self.with_scaling, quantile_range=self.quantile_range
        )
