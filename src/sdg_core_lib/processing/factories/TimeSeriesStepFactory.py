from sdg_core_lib.data_type import DataType
from sdg_core_lib.processing.PipelineConfig import PipelineStepConfig
from sdg_core_lib.processing.factories.StepFactory import (
    PipelineStepFactory,
)
from sdg_core_lib.processing.pipeline.steps.TimeSeriesScaler import (
    TimeSeriesScaler,
)


class TimeSeriesStepFactory(PipelineStepFactory):
    data_type = DataType.TIMESERIES

    @staticmethod
    def create_scaler(config: PipelineStepConfig) -> TimeSeriesScaler:
        return TimeSeriesScaler(
            data_type=TimeSeriesStepFactory.data_type, config=config
        )
