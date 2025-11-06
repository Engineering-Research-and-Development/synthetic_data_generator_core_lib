from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy

from dataset import ColumnRegistry, NumericColumn
from dataset.Processor import Processor
from dataset.columns import Column


class Dataset(ABC):

    def __init__(self, processor: Processor):
        self.processor = processor

    @classmethod
    @abstractmethod
    def from_json(cls, json_data: list[dict], processor: Processor) -> 'Dataset':
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> 'Dataset':
        raise NotImplementedError

    @abstractmethod
    def to_json(self) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def to_registry(self) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def preprocess(self) -> 'Dataset':
        raise NotImplementedError

    @abstractmethod
    def inverse_preprocess(self) -> 'Dataset':
        raise NotImplementedError




class Table(Dataset):

    def __init__(self, columns: list[Column], processor: Processor, pk_index: int= None):
        super().__init__(processor)
        self.columns = columns
        self.pk_col_index = pk_index
        self.shape = self.get_shape()

    @classmethod
    def from_json(cls, json_data: list[dict], processor: Processor) -> 'Table':
        pk_index = None
        columns = []
        for idx, col_data in enumerate(json_data):
            col_type = col_data.get("column_type", "")
            col_name = col_data.get("column_name", "")
            col_values = np.array(col_data.get("column_data", [])).T
            col_value_type = col_data.get("column_datatype", "")
            col_position = idx

            if col_type == "primary_key":
                if pk_index is not None:
                    raise ValueError(f"A primary key for dataset already exists: {columns[pk_index].name} but another primary key was found: {col_name}")
                pk_index = col_position

            col = ColumnRegistry.get_column(col_value_type)(
                col_name,
                col_value_type,
                col_position,
                col_values,
            )
            columns.append(col)

        return Table(columns, processor, pk_index)


    def clone(self) -> 'Table':
        return Table(deepcopy(self.columns), self.processor, self.pk_col_index)

    def to_json(self) -> list[dict]:
        return [
            {
                "column_data": col.values.tolist(),
                "column_name": col.name,
                "column_type": col.column_type,
                "column_datatype": col.value_type,
            }
            for col in self.columns]

    def to_registry(self) -> list[dict]:
        return [
            {
                "feature_name": col.name,
                "feature_position": col.position,
                "column_type": col.column_type,
                "type": col.value_type
            }
            for col in self.columns
        ]


    def preprocess(self) -> "Table":
        pass

    def inverse_preprocess(self) -> 'Table':
        pass

    def get_primary_key(self) -> Column:
        return self.columns[self.pk_col_index]

    def get_data(self) -> np.ndarray:
        return np.array([col.get_data() for col in self.columns]).T

    def get_shape(self) -> tuple[int, ...]:
        col_shape_total = np.sum([col.get_internal_shape()[1] for col in self.columns])
        # We assume get_shape picks the first column shape as the row shape
        row_shape_total = self.columns[0].get_internal_shape()[0]
        return row_shape_total, col_shape_total

    def get_numeric_column(self) -> list[NumericColumn]:
        return [col for col in self.columns if isinstance(col, NumericColumn)]




