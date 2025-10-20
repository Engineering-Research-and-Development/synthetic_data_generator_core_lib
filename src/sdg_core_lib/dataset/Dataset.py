from sdg_core_lib.dataset.Column import Column, ColumnMetadata
import numpy as np

from sdg_core_lib.dataset.TypedSubDataset import TypedSubDataset


class Dataset:
    def __init__(self, columns: list[Column]):
        self.columns = columns

    @classmethod
    def from_json(cls, json_data: dict):
        columns = [Column.from_json(col, idx) for idx, col in enumerate(json_data)]
        return cls(columns)

    @classmethod
    def from_data_and_metadata(cls, data: np.ndarray, metadata: list[ColumnMetadata]):
        columns = []
        for column_idx, column_metadata in metadata:
            columns.append(Column(data[:, column_idx], column_metadata))
        return cls(columns)

    @classmethod
    def from_subdatasets(cls, subdatasets: list[TypedSubDataset]):
        columns = []
        for subdataset in subdatasets:
            columns.extend(subdataset.columns)
        return cls(columns)

    def separate_into_subdatasets(self) -> list[TypedSubDataset]:
        subdatasets = []
        all_data_types = set([column.metadata.column_type for column in self.columns])
        for data_type in all_data_types:
            grouped_columns = [
                col for col in self.columns if col.metadata.column_type == data_type
            ]
            subdatasets.append(TypedSubDataset(grouped_columns, data_type))
        return subdatasets

    def to_numpy(self) -> np.ndarray:
        array_data = np.array([col.data for col in self.columns])
        return np.moveaxis(array_data, 0, 1)

    def get_metadata(self) -> list[dict]:
        return [col.metadata.to_json() for col in self.columns]

    def to_registry(self) -> list[dict]:
        return [column.to_registry() for column in self.columns]
