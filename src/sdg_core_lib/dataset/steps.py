from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Step(ABC):
    name: str = None
    operator = None

    @abstractmethod
    def save(self, directory_path: str):
        pass

    @abstractmethod
    def load(self, directory_path: str):
        pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        pass


class MinMaxWrapper(Step):
    def __init__(self):
        self.name = "minmax_scaler"
        self.operator = None

    def save(self, directory_path: str):
        pass

    def load(self, directory_path: str):
        pass

    def transform(self, data: np.ndarray) -> np.ndarray:
        self.operator = MinMaxScaler()
        return self.operator.fit_transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.operator is None:
            raise ValueError("Operator not initialized")
        return self.operator.inverse_transform(data)

