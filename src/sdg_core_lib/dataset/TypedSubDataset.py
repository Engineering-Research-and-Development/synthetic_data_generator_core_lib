import numpy as np

from sdg_core_lib.data_type import DataType
from sdg_core_lib.dataset.Column import Column, ColumnMetadata
from sdg_core_lib.dataset.Dataset import Dataset
from sdg_core_lib.processing.processors import processor_map


class TypedSubDataset(Dataset):
    def __init__(self, columns: list[Column], data_type: DataType):
        super().__init__(columns)
        self.columns = columns
        self.data_type = data_type
        self.processor = self._set_processor()

    def _set_processor(self):
        processor = processor_map.get(self.data_type)
        if not processor:
            raise ValueError(f"Unsupported data type: {self.data_type}")
        return processor

    @classmethod
    def from_json(cls, json_data: dict):
        raise NotImplementedError("from_json is not allowed in TypedSubDatasets")

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

    @classmethod
    def from_subdatasets(cls, subdatasets: list["TypedSubDataset"]):
        raise NotImplementedError("from_subdatasets is not allowed in TypedSubDatasets")

    def separate_into_subdatasets(self):
        raise NotImplementedError(
            "separate_into_subdatasets is not allowed in TypedSubDatasets"
        )
