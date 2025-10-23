from enum import Enum


class ColumnType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TIMESERIES = "time_series"
    NONE = ""


class DataType(Enum):
    INT = "int32"
    FLOAT = "float32"
    DOUBLE = "float64"
    BOOL = "bool"
    STRING = "string"
    NONE = ""
