from sdg_core_lib.data_type import DataType
from sdg_core_lib.processing.PipelineConfig import PipelineStepConfig
from sdg_core_lib.processing.factories.StepFactory import (
    PipelineStepFactory,
)
from sdg_core_lib.processing.pipeline.steps.NumericScaler import (
    NumericScaler,
)


class NumericStepFactory(PipelineStepFactory):
    data_type = DataType.NUMERIC

    @staticmethod
    def create_scaler(config: PipelineStepConfig) -> NumericScaler:
        return NumericScaler(data_type=NumericStepFactory.data_type, config=config)
