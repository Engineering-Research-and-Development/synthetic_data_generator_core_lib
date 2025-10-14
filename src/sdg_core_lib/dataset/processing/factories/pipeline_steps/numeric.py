from sdg_core_lib.dataset.processing.base.pipeline_step_factory import (
    PipelineStepFactory,
)
from sdg_core_lib.dataset.processing.processors.numeric.steps.scaler import (
    NumericScaler,
)


class NumericStepFactory(PipelineStepFactory):
    @staticmethod
    def create_scaler(mode: str) -> NumericScaler:
        return NumericScaler(mode)
