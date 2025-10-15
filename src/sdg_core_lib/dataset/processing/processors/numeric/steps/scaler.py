import numpy as np
from typing import Tuple

from sdg_core_lib.dataset.processing.base.pipeline_steps.scaler import (
    Scaler,
)


class NumericScaler(Scaler):
    """
    A Scaler implementation for numerical data that applies feature-wise scaling.

    This scaler is designed for 2D numerical data where each column represents a feature
    and each row represents a sample. It supports both Min-Max and Standard scaling
    strategies through its parent Scaler class.

    The scaler can be used as part of a processing pipeline to normalize or standardize
    numerical features, which is particularly useful for machine learning algorithms that
    are sensitive to the scale of input features.

    Example:
        scaler = NumericScaler(mode='standard')
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        X_scaled, _ = scaler.fit_transform(X_train)
    """

    def __init__(self, data_type: str, mode: str) -> None:
        """
        Initialize the NumericScaler with the specified scaling mode.

        Args:
            data_type: The type of data to be scaled.
            mode: The scaling strategy to use. Must be one of:
                - 'minmax': Uses scikit-learn's MinMaxScaler
                - 'standard': Uses scikit-learn's StandardScaler

        Raises:
            ValueError: If an invalid mode is provided.
        """
        super().__init__(data_type, mode)

    def _pre_process(self, data: np.ndarray) -> np.ndarray:
        """
        Pre-process the data before scaling.

        For NumericScaler, this is a pass-through method that returns the input data
        as-is, since no special processing is needed for 2D numerical data.

        Args:
            data: Input data to be pre-processed. Should be a 2D numpy array.

        Returns:
            The input data unchanged.

        Raises:
            ValueError: If the input data is not a 2D array.
        """
        if len(data.shape) != 2:
            raise ValueError(f"Expected 2D input data, got shape {data.shape}")
        return data

    def _post_process(
        self, data: np.ndarray, original_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Post-process the data after scaling.

        For NumericScaler, this is a pass-through method that returns the scaled data
        as-is, since no special post-processing is needed for 2D numerical data.

        Args:
            data: Scaled data to be post-processed.
            original_shape: The shape of the original data before pre-processing.

        Returns:
            The scaled data unchanged.
        """
        return data
