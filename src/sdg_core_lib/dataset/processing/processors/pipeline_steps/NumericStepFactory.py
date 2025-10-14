from sdg_core_lib.dataset.processing.processors.PipelineStepFactory import (
    PipelineStepFactory,
)
from sdg_core_lib.dataset.processing.processors.pipeline_steps.scalers.NumericScaler import (
    NumericScaler,
)


class NumericStepFactory(PipelineStepFactory):
    @staticmethod
    def create_scaler(mode: str) -> NumericScaler:
        return NumericScaler(mode)
