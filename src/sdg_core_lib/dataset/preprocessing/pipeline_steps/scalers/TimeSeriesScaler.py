import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sdg_core_lib.dataset.preprocessing.pipeline_steps.scalers.Scaler import Scaler


class TimeSeriesScaler(Scaler):
    def __init__(self, scaler: MinMaxScaler | StandardScaler):
        super().__init__(scaler)

    def _pre_process(self, data: np.ndarray) -> np.ndarray:
        batch, features, steps = data.shape
        data_reshaped = data.transpose(0, 2, 1).reshape(-1, features)
        return data_reshaped

    def _post_process(self, data: np.ndarray, original_shape: tuple) -> np.ndarray:
        batch, features, steps = original_shape
        data = data.reshape(batch, steps, features).transpose(0, 2, 1)
        return data
