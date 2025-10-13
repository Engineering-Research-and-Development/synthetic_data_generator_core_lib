import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sdg_core_lib.dataset.preprocessing.pipeline_steps.scalers.Scaler import Scaler


class NumericScaler(Scaler):
    def __init__(self, scaler: MinMaxScaler | StandardScaler):
        super().__init__(scaler)

    def _pre_process(self, data: np.ndarray) -> np.ndarray:
        return data

    def _post_process(self, data: np.ndarray, original_shape: tuple) -> np.ndarray:
        return data
