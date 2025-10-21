from sdg_core_lib.dataset.Column import Column, ColumnMetadata
import numpy as np

from sdg_core_lib.dataset.DatasetComponent import DatasetComponent
from sdg_core_lib.dataset.TypedSubDataset import TypedSubDataset


class Dataset(DatasetComponent):
    def __init__(self, columns: list[Column]):
        super().__init__(columns)

    @classmethod
    def from_json(cls, json_data: list[dict]):
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
