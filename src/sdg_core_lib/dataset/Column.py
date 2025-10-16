import numpy as np
from enum import Enum

class ColumnType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TIMESERIES = "timeseries"
    NONE = ""

class ColumnMetadata:
    def __init__(self, name: str, column_type: str, data_type: str):
        self.name = name
        self.column_type = ColumnType(column_type)
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

