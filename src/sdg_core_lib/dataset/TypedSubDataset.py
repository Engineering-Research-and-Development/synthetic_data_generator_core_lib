import numpy as np

from sdg_core_lib.types import ColumnType
from sdg_core_lib.dataset.Column import Column, ColumnMetadata
from sdg_core_lib.dataset.DatasetComponent import DatasetComponent


class TypedSubDataset(DatasetComponent):
    def __init__(self, columns: list[Column], column_type: ColumnType):
        super().__init__(columns)
        self.column_type = column_type

    @classmethod
    def from_data_and_metadata(cls, data: np.ndarray, metadata: list[ColumnMetadata]):
        data_types = [col_metadata.column_type for col_metadata in metadata]
        if len(set(data_types)) > 1:
            raise ValueError(
                "TypedSubDataset can only be created from data with a single data type"
            )
        data_type = data_types[0]
        columns = []
        for column_idx, column_metadata in enumerate(metadata):
            columns.append(Column(data[:, column_idx], column_metadata))
        return cls(columns, data_type)

    def get_processing_shape(self) -> str:
        return str(self.to_numpy().shape[1:])
