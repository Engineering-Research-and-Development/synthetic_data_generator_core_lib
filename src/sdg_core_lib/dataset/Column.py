import numpy as np

from sdg_core_lib.data_type import DataType


class ColumnMetadata:
    def __init__(self, name: str, column_type: str, data_type: str, position: int):
        self.name = name
        self.position = position
        self.column_type = DataType(column_type)
        self.data_type = data_type

    def to_json(self):
        return {
            "column_name": self.name,
            "column_type": self.column_type.value,
            "column_datatype": self.data_type,
        }


class Column:
    def __init__(self, data: np.ndarray, metadata: ColumnMetadata):
        self.data = data
        self.metadata = metadata

    @classmethod
    def from_json(cls, col_data: dict, position: int):
        return cls(
            np.array(col_data.get("column_data", [])),
            ColumnMetadata(
                col_data.get("column_name"),
                col_data.get("column_type", ""),
                col_data.get("column_datatype", ""),
                position,
            ),
        )

    def to_json(self) -> dict:
        payload = {"column_data": self.data.tolist()}
        payload.update(self.metadata.to_json())
        return payload

    def to_registry(self):
        # TODO: It is necessary to switch from "is categorical" to other fields!
        return {
            "feature_name": self.metadata.name,
            "feature_position": self.metadata.position,
            "is_categorical": True
            if self.metadata.column_type == DataType.CATEGORICAL
            else False,
            "feature_datatype": self.metadata.data_type,
        }
