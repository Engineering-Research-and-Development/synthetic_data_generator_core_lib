from abc import ABC, abstractmethod

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sdg_core_lib.dataset.processing.base.pipeline_steps.scaler import (
    Scaler,
)


class PipelineStepFactory(ABC):
    @staticmethod
    @abstractmethod
    def create_scaler(scaler: MinMaxScaler | StandardScaler) -> Scaler:
        raise NotImplementedError
