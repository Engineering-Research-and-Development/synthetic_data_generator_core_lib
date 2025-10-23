from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger

from sdg_core_lib.process.PipelineConfig import (
    PipelineConfig,
)
from sdg_core_lib.process.factories.StepFactory import (
    PipelineStepFactory,
)
from sdg_core_lib.process.pipeline.PipelineStep import (
    PipelineStep,
)


class ProcessingPipeline:
    """
    A pipeline for applying a sequence of process pipeline_steps to data.

    This class manages the execution of process pipeline_steps in sequence, handling both
    training and test data while maintaining the state of each step.
    """

    def __init__(
        self, step_factory: PipelineStepFactory, config: PipelineConfig
    ) -> None:
        """
        Initialize the process pipeline.

        Args:
            step_factory: Factory for creating process pipeline_steps.
            config: Configuration object containing pipeline parameters.
        """
        self.step_factory = step_factory
        self.config = config
        self.steps: Dict[str, PipelineStep] = {}
        self._prepare()
        self.is_all_fit = False

    def _prepare(self) -> None:
        """
        Initialize all process pipeline_steps based on the configuration.

        This method creates and configures each process step using the step factory
        and the provided configuration.
        """
        for step_name, step_config in self.config.get_full_config().items():
            try:
                self.steps[step_name] = getattr(
                    self.step_factory, f"create_{step_name}"
                )(step_config)
                logger.info(
                    f"Added {step_config.value} {step_name} to process pipeline"
                )
            except AttributeError:
                logger.info(
                    f"Skipping creation of {step_name} which is not supported by StepFactory"
                )
                pass

    def load(self, folder_path: str) -> None:
        """
        Load the state of all pipeline pipeline_steps from disk.

        Args:
            folder_path: Directory path containing the saved pipeline states.

        Raises:
            FileNotFoundError: If the specified directory or state files are not found.
            OSError: If there is an error reading from the specified directory.
        """
        for step_name in self.steps:
            self.steps[step_name].load(folder_path)
        self.is_all_fit = True

    def save(self, folder_path: str) -> None:
        """
        Save the state of all pipeline pipeline_steps to disk.

        Args:
            folder_path: Directory path where the pipeline states should be saved.

        Raises:
            OSError: If there is an error writing to the specified directory.
        """
        for step_name in self.steps:
            self.steps[step_name].save(folder_path)

    def fit_transform(
        self, train_data: np.ndarray, test_data: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply all process pipeline_steps to the input data.

        This method applies each process step in sequence to the training data,
        fitting the step if necessary, and then applies the same transformations
        to the test data without refitting.

        Args:
            train_data: The training data to preprocess.
            test_data: Optional test data to apply the same transformations to.

        Returns:
            A tuple containing the preprocessed training and test data.
        """
        for step_name in self.steps:
            train_data, test_data = self.steps[step_name].fit_transform(
                train_data, test_data
            )
        self.is_all_fit = True
        return train_data, test_data

    def transform(
        self, train_data: np.ndarray, test_data: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply all process pipeline_steps to the input data.

        This method applies each process step in sequence to the training data,
        fitting the step if necessary, and then applies the same transformations
        to the test data without refitting.

        Args:
            train_data: The training data to preprocess.
            test_data: Optional test data to apply the same transformations to.

        Returns:
            A tuple containing the preprocessed training and test data.
        """
        for step_name in self.steps:
            train_data, test_data = self.steps[step_name].transform(
                train_data, test_data
            )
        return train_data, test_data

    def inverse_transform(
        self, train_data: np.ndarray, test_data: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the inverse of all process pipeline_steps to the input data.

        This method applies the inverse of each process step in reverse order
        to the training data, and then applies the inverse of the same transformations
        to the test data.

        Args:
            train_data: The training data to preprocess.
            test_data: Optional test data to apply the same transformations to.

        Returns:
            A tuple containing the preprocessed training and test data.
        """
        for step_name in reversed(self.steps):
            train_data, test_data = self.steps[step_name].inverse_transform(
                train_data, test_data
            )
        return train_data, test_data
