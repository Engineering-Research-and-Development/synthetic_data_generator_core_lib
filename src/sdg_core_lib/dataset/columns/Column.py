import pandas as pd
import numpy as np


class Column:
    def __init__(self, name: str, data: np.ndarray, data_type: str, column_type: str):
        self.name = name
        self.data = data
        self.data_type = data_type
        self.column_type = column_type

    @classmethod
    def from_json(cls, col_data: dict):
        return cls(
            col_data.get("column_name"),
            np.array(col_data.get("column_data", [])),
            col_data.get("column_type", ""),
            col_data.get("column_datatype", ""),
        )
