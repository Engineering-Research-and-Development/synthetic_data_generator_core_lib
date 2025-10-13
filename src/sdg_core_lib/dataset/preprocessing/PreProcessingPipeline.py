from abc import ABC, abstractmethod

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sdg_core_lib.dataset.preprocessing.pipeline_steps.scalers.Scaler import Scaler


class PreProcessingPipeline(ABC):

    def __init__(self):
        self.scaler: Scaler | None = None
        # self.encoder : Encoder | None = none
        # etc
        self._prepare(None)

    @abstractmethod
    def _prepare(self, scaler: MinMaxScaler | StandardScaler | None):
        raise NotImplementedError

    @abstractmethod
    def compute(self, train_data: np.ndarray, test_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

