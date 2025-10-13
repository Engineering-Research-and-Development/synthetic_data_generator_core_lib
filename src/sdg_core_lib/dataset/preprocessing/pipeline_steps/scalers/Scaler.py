from abc import ABC, abstractmethod

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Scaler(ABC):
    """
    Abstract base class for data scaling operations.

    This class provides a unified interface for scaling numerical data using different scaling methods.
    It wraps scikit-learn scalers and adds pre/post-processing capabilities.

    Attributes:
        scaler: An instance of scikit-learn's MinMaxScaler or StandardScaler
        is_fit (bool): Flag indicating whether the scaler has been fitted
    """

    def __init__(self, scaler: MinMaxScaler | StandardScaler):
        """
        Initialize the Scaler with a scikit-learn scaler.

        Args:
            scaler: An instance of scikit-learn's MinMaxScaler or StandardScaler
        """
        self.scaler = scaler
        self.is_fit = False

    @abstractmethod
    def _pre_process(self, data: np.ndarray) -> np.ndarray:
        """
        Pre-process the data before scaling.

        Args:
            data: Input data to be pre-processed

        Returns:
            Pre-processed data ready for scaling
        """
        raise NotImplementedError

    @abstractmethod
    def _post_process(self, data: np.ndarray, original_shape: tuple) -> np.ndarray:
        """
        Post-process the data after scaling.

        Args:
            data: Scaled data to be post-processed
            original_shape: original data shape

        Returns:
            Post-processed scaled data
        """
        raise NotImplementedError

    def transform(
        self, data: np.ndarray, test_data: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Scale the data using the fitted scaler.

        Args:
            data: Training data to be transformed
            test_data: Optional test data to be transformed using the same scaling

        Returns:
            Tuple of (transformed_training_data, transformed_test_data)
            If test_data is None, the second element will be None
        """
        data_shape = data.shape
        data = self._pre_process(data)
        data = self.scaler.transform(data)
        data = self._post_process(data)
        if test_data is not None:
            test_data = self._pre_process(test_data)
            test_data = self.scaler.transform(test_data)
            test_data = self._post_process(test_data, data_shape)

        return data, test_data

    def fit_transform(
        self, data: np.ndarray, test_data: np.ndarray = None
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
        data = self._post_process(data)
        self.is_fit = True
        if test_data is not None:
            test_data = self._pre_process(test_data)
            test_data = self.scaler.transform(test_data)
            test_data = self._post_process(test_data, data_shape)

        return data, test_data

    def inverse_transform(
        self, data: np.ndarray, test_data: np.ndarray = None
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
        data_shape = data.shape
        data = self._pre_process(data)
        data = self.scaler.inverse_transform(data)
        data = self._post_process(data, data_shape)
        if test_data is not None:
            test_data = self._pre_process(test_data)
            test_data = self.scaler.inverse_transform(test_data)
            test_data = self._post_process(test_data)

        return data, test_data
