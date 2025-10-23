from sdg_core_lib.data_type import DataType
from sdg_core_lib.process.PipelineConfig import ScalerConfig
from sdg_core_lib.process.factories.StepFactory import (
    PipelineStepFactory,
)
from sdg_core_lib.process.pipeline.steps.NumericScaler import (
    NumericScaler,
)


class NumericStepFactory(PipelineStepFactory):
    data_type = DataType.NUMERIC

    @staticmethod
    def create_scaler(config: ScalerConfig) -> NumericScaler:
        return NumericScaler(data_type=NumericStepFactory.data_type, config=config)
