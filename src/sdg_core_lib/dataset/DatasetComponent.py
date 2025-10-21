from sdg_core_lib.dataset.Column import Column, ColumnMetadata
import numpy as np


class DatasetComponent:
    def __init__(self, columns: list[Column]):
        self.columns = columns

    @classmethod
    def from_json(cls, json_data: list[dict]):
        raise NotImplementedError(f"This method is not implemented for class {cls.__name__}")

    @classmethod
    def from_data_and_metadata(cls, data: np.ndarray, metadata: list[ColumnMetadata]):
        raise NotImplementedError(f"This method is not implemented for class {cls.__name__}")

    @classmethod
    def from_subdatasets(cls, subdatasets: list['DatasetComponent']):
        raise NotImplementedError(f"This method is not implemented for class {cls.__name__} ")

    def separate_into_subdatasets(self) -> list['DatasetComponent']:
        raise NotImplementedError(f"This method is not implemented for class {self.__class__.__name__}")

    def to_numpy(self) -> np.ndarray:
        array_data = np.array([col.data for col in self.columns])
        return np.moveaxis(array_data, 0, 1)

    def get_metadata(self) -> list[ColumnMetadata]:
        return [col.metadata for col in self.columns]

    def to_registry(self) -> list[dict]:
        return [column.to_registry() for column in self.columns]

    def to_json(self) -> list[dict]:
        return [column.to_json() for column in self.columns]

