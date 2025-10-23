from sdg_core_lib.process.pipeline.ProcessingPipeline import (
    ProcessingPipeline,
)
from sdg_core_lib.process.processors.Processor import Processor
from sdg_core_lib.process.factories.NumericStepFactory import (
    NumericStepFactory,
)


class NumericProcessor(Processor):
    def __init__(self):
        """
        Initialize the NumericProcessor.

        This method initializes the NumericProcessor with the default configuration.

        Args:
            None

        Returns:
            None
        """
        super().__init__()

    def create_processing_pipeline(self) -> ProcessingPipeline:
        """
        Create a process pipeline for numeric data.

        This method initializes a NumericStepFactory and creates a ProcessingPipeline
        with the current configuration.

        Returns:
            A ProcessingPipeline instance configured for numeric data process.
        """
        step_factory = NumericStepFactory()
        pipeline = ProcessingPipeline(step_factory, self.config)
        return pipeline
