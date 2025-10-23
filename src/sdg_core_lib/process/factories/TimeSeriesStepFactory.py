from sdg_core_lib.data_type import DataType
from sdg_core_lib.process.PipelineConfig import ScalerConfig
from sdg_core_lib.process.factories.StepFactory import (
    PipelineStepFactory,
)
from sdg_core_lib.process.pipeline.steps.TimeSeriesScaler import (
    TimeSeriesScaler,
)


class TimeSeriesStepFactory(PipelineStepFactory):
    data_type = DataType.TIMESERIES

    @staticmethod
    def create_scaler(config: ScalerConfig) -> TimeSeriesScaler:
        return TimeSeriesScaler(
            data_type=TimeSeriesStepFactory.data_type, config=config
        )
