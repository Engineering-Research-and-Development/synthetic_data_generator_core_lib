from abc import ABC, abstractmethod

from sdg_core_lib.process.PipelineConfig import ScalerConfig
from sdg_core_lib.process.pipeline.steps.Scaler import (
    Scaler,
)


class PipelineStepFactory(ABC):
    @staticmethod
    @abstractmethod
    def create_scaler(config: ScalerConfig) -> Scaler:
        raise NotImplementedError
