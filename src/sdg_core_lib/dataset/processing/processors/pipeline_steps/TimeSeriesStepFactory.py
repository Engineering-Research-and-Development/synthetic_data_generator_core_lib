from sdg_core_lib.dataset.processing.processors.PipelineStepFactory import (
    PipelineStepFactory,
)
from sdg_core_lib.dataset.processing.processors.pipeline_steps.scalers.TimeSeriesScaler import (
    TimeSeriesScaler,
)


class TimeSeriesStepFactory(PipelineStepFactory):

    
    @staticmethod
    def create_scaler(mode: str) -> TimeSeriesScaler:
        return TimeSeriesScaler(mode)
