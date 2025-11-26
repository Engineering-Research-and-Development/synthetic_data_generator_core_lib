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

# TODO: What if steps change data types? Should I implement some "old_type / new_type" mechanics?


class Step(ABC):
    def __init__(self, type: str, position: int, col_name: str, mode: str):
        self.name = type
        self.mode = mode
        self.position = position
        self.col_name = col_name
        self.operator = None
        self.filename = f"{self.position}_{self.col_name}_{self.mode}_{self.name}.skops"

    @abstractmethod
    def _set_operator(self):
        raise NotImplementedError

    def save(self, directory_path: str):
        if self.operator is None:
            raise ValueError("Operator is not created")
        os.makedirs(directory_path, exist_ok=True)
        filename = os.path.join(directory_path, self.filename)
        if not os.path.exists(filename):
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
            raise ValueError("Operator not initialized")
        return self.operator.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.operator is None:
            raise ValueError("Operator not initialized")
        return self.operator.inverse_transform(data)


class NoneStep(Step):
    def __init__(self, position: int, mode=None):
        super().__init__(type="none", position=position, col_name="", mode=mode)

    def save(self, directory_path: str):
        pass

    def load(self, directory_path: str):
        pass

    def _set_operator(self):
        pass

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data


class ScalerWrapper(Step):
    def __init__(self, position: int, col_name: str, mode: Literal["minmax", "standard"] = "standard",):
        super().__init__(type="scaler", position=position, col_name=col_name, mode=mode)

    def _set_operator(self):
        if self.mode == "minmax":
            return MinMaxScaler()
        elif self.mode == "standard":
            return StandardScaler()
        else:
            raise ValueError("Invalid mode while setting the scaler")


class LabelEncoderWrapper(Step):
    def __init__(self, position: int, col_name: str, mode="label"):
        super().__init__(type="encoder", position=position, col_name=col_name, mode=mode)

    def _set_operator(self):
        return LabelEncoder()


class OneHotEncoderWrapper(Step):
    def __init__(self, position: int, col_name: str, mode="one_hot"):
        super().__init__(type="encoder", position=position, col_name=col_name, mode=mode)

    def _set_operator(self):
        return OneHotEncoder()

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return super().fit_transform(data).toarray()

    def transform(self, data: np.ndarray) -> np.ndarray:
        return super().transform(data).toarray()
