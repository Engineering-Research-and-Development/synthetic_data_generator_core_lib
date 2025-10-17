from abc import ABC, abstractmethod

from sdg_core_lib.processing.PipelineConfig import ScalerConfig
from sdg_core_lib.processing.pipeline.steps.Scaler import (
    Scaler,
)


class PipelineStepFactory(ABC):
    @staticmethod
    @abstractmethod
    def create_scaler(config: ScalerConfig) -> Scaler:
        raise NotImplementedError
