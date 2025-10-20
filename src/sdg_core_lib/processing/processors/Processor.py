from abc import ABC, abstractmethod
from numpy import ndarray
from typing import Optional, Tuple

from sdg_core_lib.processing.PipelineConfig import (
    PipelineConfig,
)
from sdg_core_lib.processing.pipeline.ProcessingPipeline import (
    ProcessingPipeline,
)


class Processor(ABC):
    """
    Abstract base class for all data processors in the SDG Core Library.

    This class defines the interface for processing pipelines and provides common functionality
    for executing, saving, and loading processing pipelines. Concrete implementations should
    be created for different data types (e.g., numeric, time series).
    """

    def __init__(self) -> None:
        """
        Initialize the Preprocessor with a configuration object.

        Args:
            config: A PipelineConfig object containing the configuration for the processing pipeline.
        """
        self.config = None
        self.pipeline: Optional[ProcessingPipeline] = None

    def execute_preprocessing(
        self, data: ndarray, test_data: ndarray
    ) -> Tuple[ndarray, ndarray]:
        """
        Execute the processing pipeline on the provided data.

        Args:
            data: The training data to be preprocessed.
            test_data: Optional test data to be preprocessed using the same transformations.

        Returns:
            A tuple containing the preprocessed training data and test data.
            If test_data is None, the second element will be None.

        Raises:
            AttributeError: If the pipeline has not been initialized.
        """
        if self.pipeline is None:
            raise AttributeError(
                "Pipeline has not been initialized. Call create_processing_pipeline() first."
            )

        if not self.pipeline.is_all_fit:
            train_data_preprocessed, test_data_preprocessed = (
                self.pipeline.fit_transform(train_data=data, test_data=test_data)
            )
        else:
            train_data_preprocessed, test_data_preprocessed = self.pipeline.transform(
                train_data=data, test_data=test_data
            )
        return train_data_preprocessed, test_data_preprocessed

    def execute_postprocessing(
        self, train_data_preprocessed: ndarray, test_data_preprocessed: ndarray
    ):
        train_data, test_data = self.pipeline.inverse_transform(
            train_data_preprocessed, test_data_preprocessed
        )
        return train_data, test_data

    def save_pipeline(self, folder_path: str) -> None:
        """
        Save the current processing pipeline to disk.

        Args:
            folder_path: Directory path where the pipeline should be saved.

        Raises:
            AttributeError: If the pipeline has not been initialized.
            OSError: If there is an error writing to the specified directory.
        """
        if self.pipeline is None:
            raise AttributeError("Pipeline is not instantiated")
        self.pipeline.save(folder_path)

    def load_pipeline(self, folder_path: str) -> None:
        """
        Load a processing pipeline from disk.

        Args:
            folder_path: Directory path containing the saved pipeline.

        Raises:
            FileNotFoundError: If the specified directory or pipeline files are not found.
            OSError: If there is an error reading from the specified directory.
        """
        if self.config is None:
            raise AttributeError("Config is not instantiated. Call set_config() first.")
        if self.pipeline is None:
            self.pipeline = self.create_processing_pipeline()
        self.pipeline.load(folder_path)

    def set_config(self, config: PipelineConfig):
        self.config = config
        self.pipeline = self.create_processing_pipeline()
        return self

    @staticmethod
    @abstractmethod
    def create_processing_pipeline() -> ProcessingPipeline:
        """
        Create and return a new processing pipeline.

        This method must be implemented by subclasses to create a specific type of
        processing pipeline based on the data type being processed.

        Returns:
            A ProcessingPipeline instance configured for the specific data type.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement create_processing_pipeline"
        )
