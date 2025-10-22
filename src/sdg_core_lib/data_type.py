from enum import Enum


class DataType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TIMESERIES = "time_series"
    NONE = ""
