import numpy as np
from typing import Tuple

from sdg_core_lib.types import ColumnType
from sdg_core_lib.process.PipelineConfig import ScalerConfig
from sdg_core_lib.process.pipeline.steps.Scaler import (
    Scaler,
)


class TimeSeriesScaler(Scaler):
    """
    A Scaler implementation for time series data that applies feature-wise scaling.

    This scaler is specifically designed for 3D time series data with shape
    (batch_size, n_features, n_timesteps). It handles the reshaping of the data
    to work with scikit-learn's scalers, which expect 2D input.

    The scaler can be used to normalize or standardize time series data while
    preserving the temporal structure. It's particularly useful for deep learning
    models that expect input in the form of (samples, timesteps, features).

    Example:
        scaler = TimeSeriesScaler(mode='standard')
        # X_train shape: (batch_size=3, n_features=2, n_timesteps=4)
        X_train = np.random.rand(3, 2, 4)
        X_scaled, _ = scaler.fit_transform(X_train)
    """

    def __init__(self, column_type: ColumnType, config: ScalerConfig) -> None:
        """
        Initialize the TimeSeriesScaler with the specified scaling mode.

        Args:
            column_type: The type of data to be scaled.
            mode: The scaling strategy to use. Must be one of:
                - 'minmax': Uses scikit-learn's MinMaxScaler
                - 'standard': Uses scikit-learn's StandardScaler

        Raises:
            ValueError: If an invalid mode is provided.
        """
        super().__init__(column_type, config)

    def _pre_process(self, data: np.ndarray) -> np.ndarray:
        """
        Reshape the time series data for scaling.

        This method reshapes the input data from (batch_size, n_features, n_timesteps)
        to (batch_size * n_timesteps, n_features) to make it compatible with
        scikit-learn's scalers.

        Args:
            data: Input time series data with shape (batch_size, n_features, n_timesteps).

        Returns:
            Reshaped 2D array with shape (batch_size * n_timesteps, n_features).

        Raises:
            ValueError: If the input data is not a 3D array.
        """
        if len(data.shape) != 3:
            raise ValueError(f"Expected 3D input data, got shape {data.shape}")

        batch, features, steps = data.shape
        # Reshape to (batch * pipeline_steps, features) for scaling
        data_reshaped = data.transpose(0, 2, 1).reshape(-1, features)
        return data_reshaped

    def _post_process(
        self, data: np.ndarray, original_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Reshape the scaled data back to the original time series format.

        This method reshapes the scaled data from (batch_size * n_timesteps, n_features)
        back to the original shape (batch_size, n_features, n_timesteps).

        Args:
            data: Scaled data with shape (batch_size * n_timesteps, n_features).
            original_shape: The original shape of the data as (batch_size, n_features, n_timesteps).

        Returns:
            Reshaped 3D array with the original shape (batch_size, n_features, n_timesteps).
        """
        batch, features, steps = original_shape
        # Reshape back to (batch, pipeline_steps, features) and then to (batch, features, time_steps)
        data = data.reshape(batch, steps, features).transpose(0, 2, 1)
        return data
