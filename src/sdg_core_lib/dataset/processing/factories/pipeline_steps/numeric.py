from sdg_core_lib.dataset.processing.base.pipeline_step_factory import (
    PipelineStepFactory,
)
from sdg_core_lib.dataset.processing.processors.numeric.steps.scaler import (
    NumericScaler,
)


class NumericStepFactory(PipelineStepFactory):
    data_type = "numeric"
    @staticmethod
    def create_scaler(mode: str) -> NumericScaler:
        return NumericScaler(data_type=NumericStepFactory.data_type, mode=mode)
