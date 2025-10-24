from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import skops.io as sio
import os

from sdg_core_lib.types import ColumnType
from sdg_core_lib.process.PipelineConfig import ScalerConfig, PipelineStepConfig
from sdg_core_lib.process.pipeline.PipelineStep import (
    PipelineStep,
)


class Scaler(PipelineStep):
    """
    A concrete implementation of PipelineStep for feature scaling operations.

    This class provides scaling functionality using scikit-learn's scalers (MinMaxScaler and StandardScaler).
    It handles both fitting the scaler to training data and applying the same scaling to test data.

    The class supports saving and loading the scaler's state to/from disk using the skops library.

    Attributes:
        scaler: The underlying scikit-learn scaler instance (MinMaxScaler or StandardScaler)
    """

    def __init__(self, column_type: ColumnType, config: PipelineStepConfig) -> None:
        """
        Initialize the Scaler with the specified scaling mode.

        Args:
            mode: The scaling strategy to use. Must be one of:
                - 'minmax': Uses scikit-learn's MinMaxScaler
                - 'standard': Uses scikit-learn's StandardScaler

        Raises:
            ValueError: If an invalid mode is provided.
        """
        super().__init__(column_type)
        self.scaler = None
        self._set_scaler(config)
        self.is_fit = False
        self._filename = f"scaler_{self.data_type.value}.skops"

    def _set_scaler(self, mode: ScalerConfig) -> None:
        """
        Initialize the appropriate scikit-learn scaler based on the specified mode.

        Args:
            mode: The scaling strategy to use ('minmax' or 'standard').

        Raises:
            ValueError: If an invalid mode is provided.
        """
        if self.scaler is not None:
            return

        if mode == ScalerConfig.MINMAX:
            self.scaler = MinMaxScaler()
        elif mode == ScalerConfig.STANDARD:
            self.scaler = StandardScaler()
        else:
            raise ValueError(
                f"Invalid scaler mode: {mode}. Must be one of {ScalerConfig.get_possible_values()}"
            )

    def save(self, folder_path: str) -> None:
        """
        Save the scaler's state to disk.

        The scaler is saved using the skops library to a file named 'scaler.skops'
        in the specified directory.

        Args:
            folder_path: Directory path where the scaler state should be saved.

        Raises:
            OSError: If there is an error creating the directory or writing the file.
        """
        os.makedirs(folder_path, exist_ok=True)
        scaler_filename = os.path.join(folder_path, self._filename)
        sio.dump(self.scaler, scaler_filename)

    def load(self, folder_path: str) -> None:
        """
        Load the scaler's state from disk.

        The scaler is loaded from a file named 'scaler.skops' in the specified directory.

        Args:
            folder_path: Directory path containing the saved scaler state.

        Raises:
            FileNotFoundError: If the scaler file is not found.
            OSError: If there is an error reading the file.
        """
        scaler_filename = os.path.join(folder_path, self._filename)
        if not os.path.isfile(scaler_filename):
            raise FileNotFoundError(f"Scaler file not found: {scaler_filename}")
        self.scaler = sio.load(scaler_filename)
        self.is_fit = True

    def transform(
        self, data: np.ndarray, test_data: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Scale the data using the fitted scaler.

        Args:
            data: Training data to be transformed. Should be a 2D numpy array.
            test_data: Optional test data to be transformed using the same scaling.
                     If provided, must have the same number of features as the training data.

        Returns:
            A tuple containing:
                - Transformed training data
                - Transformed test data (or None if test_data was None)

        Raises:
            ValueError: If the input data has a different number of features than
                      the data used to fit the scaler.
        """
        if self.scaler is None:
            raise RuntimeError("Scaler has not been initialized")
        if not self.is_fit:
            raise RuntimeError("Scaler has to be fit first")

        data_shape = data.shape
        data = self._pre_process(data)
        data = self.scaler.transform(data)
        data = self._post_process(data, data_shape)

        test_data_transformed = None
        if test_data is not None:
            test_data_shape = test_data.shape
            test_data = self._pre_process(test_data)
            test_data_transformed = self.scaler.transform(test_data)
            test_data_transformed = self._post_process(
                test_data_transformed, test_data_shape
            )

        return data, test_data_transformed

    def fit(self, data: np.ndarray) -> None:
        """
        Fit the scaler without applying transformation
        """
        preprocessed_data = self._pre_process(data)
        self.scaler.fit(preprocessed_data)
        self.is_fit = True

    def fit_transform(
        self, data: np.ndarray, test_data: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the scaler to the data and then transform it.

        Args:
            data: Training data to fit and transform
            test_data: Optional test data to be transformed using the same scaling

        Returns:
            Tuple of (fitted_and_transformed_training_data, transformed_test_data)
            If test_data is None, the second element will be None
        """
        data_shape = data.shape
        data = self._pre_process(data)
        data = self.scaler.fit_transform(data)
        self.is_fit = True
        data = self._post_process(data, data_shape)
        if test_data is not None:
            test_data_shape = test_data.shape
            test_data = self._pre_process(test_data)
            test_data = self.scaler.transform(test_data)
            test_data = self._post_process(test_data, test_data_shape)

        return data, test_data

    def inverse_transform(
        self, data: np.ndarray, test_data: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform data back to the original representation.

        Args:
            data: Transformed data to be inverted
            test_data: Optional test data to be inverted using the same scaling

        Returns:
            Tuple of (inverted_training_data, inverted_test_data)
            If test_data is None, the second element will be None
        """
        if self.scaler is None:
            raise RuntimeError("Scaler has not been initialized")
        if not self.is_fit:
            raise RuntimeError("Scaler has to be fit first")

        data_shape = data.shape
        data = self._pre_process(data)
        data = self.scaler.inverse_transform(data)
        data = self._post_process(data, data_shape)
        if test_data is not None:
            test_data_shape = test_data.shape
            test_data = self._pre_process(test_data)
            test_data = self.scaler.inverse_transform(test_data)
            test_data = self._post_process(test_data, test_data_shape)
        return data, test_data

    @abstractmethod
    def _pre_process(self, data: np.ndarray) -> np.ndarray:
        """
        Pre-process the data before applying the scaling transformation.

        This method is called before the scaler's transform method. It can be used
        for input validation, type conversion, or other process pipeline_steps.

        Args:
            data: Input data to be pre-processed. Should be a numpy array.

        Returns:
            Pre-processed data ready for scaling.

        Raises:
            ValueError: If the input data is not in the expected format.
        """
        raise NotImplementedError("Subclass have to implement _pre_process method")

    @abstractmethod
    def _post_process(
        self, data: np.ndarray, original_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Post-process the data after applying the scaling transformation.

        This method is called after the scaler's transform method. It can be used
        for reshaping, type conversion, or other post-process pipeline_steps.

        Args:
            data: Scaled data to be post-processed.
            original_shape: The shape of the original data before pre-process.

        Returns:
            Post-processed scaled data.
        """
        raise NotImplementedError("Subclass have to implement _post_process method")
