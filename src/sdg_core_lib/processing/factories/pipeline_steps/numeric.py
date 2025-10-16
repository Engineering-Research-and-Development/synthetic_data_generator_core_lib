from sdg_core_lib.processing.base.step_factory import (
    PipelineStepFactory,
)
from sdg_core_lib.processing.processors.numeric.steps.scaler import (
    NumericScaler,
)


class NumericStepFactory(PipelineStepFactory):
    data_type = "numeric"
    @staticmethod
    def create_scaler(mode: str) -> NumericScaler:
        return NumericScaler(data_type=NumericStepFactory.data_type, mode=mode)
