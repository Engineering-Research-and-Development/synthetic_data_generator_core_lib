from sdg_core_lib.dataset.Column import Column, ColumnMetadata
import numpy as np

from sdg_core_lib.dataset.DatasetComponent import DatasetComponent


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
