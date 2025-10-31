from typing import Literal

import numpy as np
from abc import ABC

from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Column:
    def __init__(self, name: str, column_type: str, data_type: str):
        self.name = name
        self.column_type = column_type
        self.data_type = data_type
        self._scaler = None
        self._encoder = None

    def scale(self, mode: Literal["standard", "minmax"]):
        raise NotImplementedError

    def inverse_scale(self):
        raise NotImplementedError

    def save_scaler(self):
        raise NotImplementedError

    def load_scaler(self):
        raise NotImplementedError


    def encode(self, mode: Literal["onehot"]):
        raise NotImplementedError

    def decode(self):
        raise NotImplementedError

    def save_encoder(self):
        raise NotImplementedError

    def load_encoder(self):
        raise NotImplementedError



class NumericColumn(Column):
    def __init__(self, name: str, column_type: str, data_type: str, values: np.ndarray):
        super().__init__(name, column_type, data_type)
        self.values = values

    def scale(self, mode: Literal["standard", "minmax"]):
        if mode == "standard":
            scaler = StandardScaler()
        elif mode == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaling mode")

        self._scaler = scaler
        scaled_values = scaler.fit_transform(self.values)
        return scaled_values

    def inverse_scale(self):
        raise NotImplementedError

    def save_scaler(self):
        raise NotImplementedError

    def load_scaler(self):
        raise NotImplementedError