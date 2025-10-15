from sdg_core_lib.dataset.processing.base.pipeline_step_factory import (
    PipelineStepFactory,
)
from sdg_core_lib.dataset.processing.processors.timeseries.steps.scaler import (
    TimeSeriesScaler,
)


class TimeSeriesStepFactory(PipelineStepFactory):
    data_type = "timeseries"
    @staticmethod
    def create_scaler(mode: str) -> TimeSeriesScaler:
        return TimeSeriesScaler(data_type=TimeSeriesStepFactory.data_type, mode=mode)
