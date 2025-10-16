from abc import ABC, abstractmethod
from sdg_core_lib.processing.base.pipeline_steps.scaler import (
    Scaler,
)


class PipelineStepFactory(ABC):
    @staticmethod
    @abstractmethod
    def create_scaler(mode: str) -> Scaler:
        raise NotImplementedError
