from typing import Optional

from sdg_core_lib.process.PipelineConfig import (
    PipelineConfig,
)
from sdg_core_lib.process.pipeline.ProcessingPipeline import (
    ProcessingPipeline,
)
from sdg_core_lib.process.processors.Processor import Processor
from sdg_core_lib.process.factories.TimeSeriesStepFactory import (
    TimeSeriesStepFactory,
)


class TimeSeriesProcessor(Processor):
    """
    A preprocessor implementation for time series data.

    This class provides a concrete implementation of the Preprocessor abstract base class
    specifically designed for time series data. It creates and manages a process
    pipeline configured with pipeline_steps appropriate for time series features.

    The preprocessor is designed to handle 3D time series data with shape
    (batch_size, n_features, n_timesteps) and supports various scaling strategies
    configured through PipelineConfig.

    Example:
        # Create a preprocessor with default configuration (no scaling)
        preprocessor = TimeSeriesPreprocessor()

        # Create a preprocessor with MinMax scaling
        config = PipelineConfig(scaler="minmax")
        preprocessor = TimeSeriesPreprocessor(config)

        # Fit and transform time series data
        # X_train shape: (batch_size=3, n_features=2, n_timesteps=4)
        X_train = np.random.rand(3, 2, 4)
        X_scaled, _ = preprocessor.execute_preprocessing_pipeline(X_train, None)
    """

    def __init__(self):
        """
        Initialize the TimeSeriesPreprocessor.
        """
        super().__init__()

    def create_processing_pipeline(self) -> ProcessingPipeline:
        """
        Create and return a process pipeline configured for time series data.

        This method initializes a TimeSeriesStepFactory and creates a PreProcessingPipeline
        with the current configuration.

        Returns:
            A PreProcessingPipeline instance configured for time series data process.
        """
        step_factory = TimeSeriesStepFactory()
        pipeline = ProcessingPipeline(step_factory, self.config)
        return pipeline
