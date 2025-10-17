import numpy as np

from sdg_core_lib.data_type import DataType


class ColumnMetadata:
    def __init__(self, name: str, column_type: str, data_type: str):
        self.name = name
        self.column_type = DataType(column_type)
        self.data_type = data_type


class Column:
    def __init__(self, data: np.ndarray, metadata: ColumnMetadata):
        self.data = data
        self.metadata = metadata

    @classmethod
    def from_json(cls, col_data: dict):
        return cls(
            np.array(col_data.get("column_data", [])),
            ColumnMetadata(
                col_data.get("column_name"),
                col_data.get("column_type", ""),
                col_data.get("column_datatype", ""),
            ),
        )
