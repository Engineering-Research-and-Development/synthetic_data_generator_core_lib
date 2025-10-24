from sdg_core_lib.types import ColumnType
from sdg_core_lib.process.PipelineConfig import ScalerConfig
from sdg_core_lib.process.factories.StepFactory import (
    PipelineStepFactory,
)
from sdg_core_lib.process.pipeline.steps.NumericScaler import (
    NumericScaler,
)


class NumericStepFactory(PipelineStepFactory):
    data_type = ColumnType.NUMERIC

    @staticmethod
    def create_scaler(config: ScalerConfig) -> NumericScaler:
        return NumericScaler(column_type=NumericStepFactory.data_type, config=config)
