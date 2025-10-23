from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from sdg_core_lib.types import ColumnType
from sdg_core_lib.process.PipelineConfig import PipelineStepConfig


class PipelineStep(ABC):
    """
    Abstract base class for all process pipeline pipeline_steps.

    This class defines the interface that all process pipeline_steps must implement.
    Each step in the process pipeline should inherit from this class and
    implement all abstract methods.
    """

    def __init__(
        self,
        data_type: ColumnType = ColumnType.NONE,
        config: PipelineStepConfig = None,
    ):
        """
        Initialize the PipelineStep with the specified data type and configuration.

        Args:
            data_type: The type of data to be processed by the step.
            config: The configuration object containing step-specific parameters.

        Note:
            If config is None, the step should use default parameters.
        """
        self.data_type = data_type
        self.config = config

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """
        Fit the process step to the training data.

        Args:
            data: The training data to fit the step on.
                Should be a numpy array of shape (n_samples, n_features).

        Note:
            This method should be implemented to compute any statistics or parameters
            needed for the transformation step.
        """
        raise NotImplementedError("Subclasses must implement fit method")

    @abstractmethod
    def transform(
        self, data: np.ndarray, test_data: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply the transformation to the data.

        Args:
            data: The data to transform. Should have the same number of features
                as the data used to fit the step.
            test_data: The test data to fit and transform


        Returns:
            The transformed data as a numpy array.

        Note:
            The transform method should not modify the input data in place.
            It should return a new array with the transformed data.
        """
        raise NotImplementedError("Subclasses must implement transform method")

    @abstractmethod
    def fit_transform(
        self, data: np.ndarray, test_data: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit the step to the data and then transform it.

        This is equivalent to calling fit() followed by transform(), but may be
        more efficient for some implementations.

        Args:
            data: The data to fit and transform.
            test_data: The test data to fit and transform

        Returns:
            The transformed data as a numpy array.
        """
        raise NotImplementedError("Subclasses must implement fit_transform method")

    @abstractmethod
    def inverse_transform(
        self, data: np.ndarray, test_data: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply the inverse transformation to the data.

        Args:
            data: The transformed data to invert.
            test_data: The transformed test data to invert if any

        Returns:
            The original data before transformation.

        Note:
            This method should be implemented if the transformation is invertible.
            For non-invertible transformations, this method should raise a NotImplementedError.
        """
        raise NotImplementedError("Subclasses must implement inverse_transform method")

    @abstractmethod
    def _pre_process(self, data: np.ndarray) -> np.ndarray:
        """
        Pre-process the data before applying the main transformation.

        This method can be used for tasks like input validation, reshaping, or
        other data preparation pipeline_steps that should happen before the main transform.

        Args:
            data: The input data to pre-process.

        Returns:
            The pre-processed data.
        """
        raise NotImplementedError("Subclasses must implement _pre_process method")

    @abstractmethod
    def _post_process(
        self, data: np.ndarray, original_shape: tuple[int, ...]
    ) -> np.ndarray:
        """
        Post-process the data after applying the main transformation.

        This method can be used for tasks like reshaping, type conversion, or
        other data cleanup pipeline_steps that should happen after the main transform.

        Args:
            data: The transformed data to post-process.
            original_shape: The shape of data before transformation

        Returns:
            The post-processed data.
        """
        raise NotImplementedError("Subclasses must implement _post_process method")

    @abstractmethod
    def save(self, folder_path: str) -> None:
        """
        Save the step's state to disk.

        Args:
            folder_path: Directory path where the step's state should be saved.

        Raises:
            OSError: If there is an error writing to the specified directory.
        """
        raise NotImplementedError("Subclasses must implement save method")

    @abstractmethod
    def load(self, folder_path: str) -> None:
        """
        Load the step's state from disk.

        Args:
            folder_path: Directory path containing the step's saved state.

        Raises:
            FileNotFoundError: If the specified directory or state file is not found.
            OSError: If there is an error reading from the specified directory.
        """
        raise NotImplementedError("Subclasses must implement load method")
