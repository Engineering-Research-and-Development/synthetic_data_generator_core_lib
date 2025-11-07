from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
)
import os
import skops.io as sio


class Step(ABC):
    def __init__(self, name: str, position: int, mode: str):
        self.name = name
        self.mode = mode
        self.position = position
        self.operator = None
        self.filename = f"{self.position}_{self.name}.skops"

    @abstractmethod
    def _set_operator(self):
        raise NotImplementedError

    def save(self, directory_path: str):
        if not self.operator:
            raise ValueError("Scaler is not created")
        os.makedirs(directory_path, exist_ok=True)
        filename = os.path.join(directory_path, self.filename)
        sio.dump(self.operator, filename)

    def load(self, directory_path: str):
        filename = os.path.join(directory_path, self.filename)
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Operator file not found: {filename}")
        self.operator = sio.load(filename)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.operator = self._set_operator()
        return self.operator.fit_transform(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.operator is None:
            raise ValueError("Scaler not initialized")
        return self.operator.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.operator is None:
            raise ValueError("Scaler not initialized")
        return self.operator.inverse_transform(data)


class ScalerWrapper(Step):
    def __init__(self, position: int, mode: Literal["minmax", "standard"] = "standard"):
        super().__init__(name="scaler", position=position, mode=mode)
        self.operator = None

    def _set_operator(self):
        if self.mode == "minmax":
            return MinMaxScaler()
        elif self.mode == "standard":
            return StandardScaler()
        else:
            raise ValueError("Invalid mode while setting the scaler")


class LabelEncoderWrapper(Step):
    def __init__(self, position: int, mode = None):
        super().__init__(name="encoder", position=position, mode=mode)

    def _set_operator(self):
        return LabelEncoder()


class OneHotEncoderWrapper(Step):
    def __init__(self, position: int, mode = None):
        super().__init__(name="encoder", position=position, mode=mode)

    def _set_operator(self):
        return OneHotEncoder()

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.operator = self._set_operator()
        return self.operator.fit_transform(data).toarray()

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.operator is None:
            raise ValueError("Scaler not initialized")
        return self.operator.transform(data).toarray()
