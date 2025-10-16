from sdg_core_lib.dataset.Column import Column, ColumnMetadata
from numpy import ndarray

class Dataset:

    def __init__(self, columns: list[Column]):
        self.columns = columns

    def _append_column(self, column: Column):
        self.columns.append(column)

    @classmethod
    def from_json(cls, json_data: dict):
        columns = [Column.from_json(col) for col in json_data]
        return cls(columns)

    @classmethod
    def from_data_and_metadata(cls, data: ndarray, metadata: list[ColumnMetadata]):
        dataset = cls([])
        for column_idx, column_metadata in metadata:
            dataset._append_column(Column(data[:, column_idx], column_metadata))
        return dataset