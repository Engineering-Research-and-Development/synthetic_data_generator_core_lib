from enum import Enum


class AllowedData:
    def __init__(self, dtype: "DataType", is_categorical: bool):
        self.dtype = dtype
        self.is_categorical = is_categorical

    def to_json(self):
        return {"type": self.dtype.value, "is_categorical": self.is_categorical}


class DataType(Enum):
    int32 = "int32"
    int64 = "int64"
    float32 = "float32"
    float64 = "float64"
    string = "str"
    bool = "bool"
