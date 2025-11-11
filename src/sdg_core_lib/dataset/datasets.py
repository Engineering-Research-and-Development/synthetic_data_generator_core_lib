from abc import ABC, abstractmethod
import numpy as np

from sdg_core_lib.dataset.columns import NumericColumn, CategoricalColumn, PrimaryKeyColumn, Column, ComputingColumn
from sdg_core_lib.dataset.processor import Processor, TableProcessor


class Dataset(ABC):

    def __init__(self, processor: Processor):
        self.processor = processor

    @classmethod
    @abstractmethod
    def from_json(cls, json_data: list[dict], processor: Processor) -> 'Dataset':
        raise NotImplementedError

    @abstractmethod
    def clone(self, new_data: np.ndarray) -> 'Dataset':
        raise NotImplementedError

    @abstractmethod
    def get_data(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_shape(self) -> tuple[int, ...]:
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
    def postprocess(self) -> 'Dataset':
        raise NotImplementedError




class Table(Dataset):

    col_registry = {
        "continuous": NumericColumn,
        "categorical": CategoricalColumn,
        "primary_key": PrimaryKeyColumn,
    }

    def __init__(self, columns: list[Column], processor: TableProcessor, pk_index: int= None):
        super().__init__(processor)
        self.columns = columns
        self.pk_col_index = pk_index
        self.shape = self.get_shape()


    @classmethod
    def from_json(cls, json_data: list[dict], processor: TableProcessor) -> 'Table':
        pk_index = None
        columns = []
        for idx, col_data in enumerate(json_data):
            col_type = col_data.get("column_type", "")
            col_name = col_data.get("column_name", "")
            col_values = np.array(col_data.get("column_data", [])).reshape(-1, 1)
            col_value_type = col_data.get("column_datatype", "")
            col_position = idx

            if col_type == "primary_key":
                if pk_index is not None:
                    raise ValueError(f"A primary key for dataset already exists: {columns[pk_index].name} but another primary key was found: {col_name}")
                pk_index = col_position

            col = cls.col_registry.get(col_type, None)(
                col_name,
                col_value_type,
                col_position,
                col_values,
            )
            columns.append(col)

        return Table(columns, processor, pk_index)

    def clone(self, data: np.ndarray) -> 'Table':
        if self.get_shape()[1] != data.shape[1]:
            raise ValueError("Data does not match table shape on column axis")
        new_columns = []
        data_idx = 0
        for col in self.columns:
            if isinstance(col, ComputingColumn):
                col_shape = col.get_internal_shape()[1]
                new_columns.append(
                    type(col)(
                        col.name,
                        col.value_type,
                        col.position,
                        data[:, data_idx:data_idx+col_shape],
                    )
                )
                data_idx += col_shape
            else:
                new_columns.append(col)
        return Table(new_columns, self.processor, self.pk_col_index)

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

    def get_primary_key(self) -> PrimaryKeyColumn:
        column =  self.columns[self.pk_col_index]
        if not isinstance(column, PrimaryKeyColumn):
            raise ValueError(f"Column {column.name} is not a primary key")
        return column

    def get_data(self) -> np.ndarray:
        return np.hstack([col.get_data() for col in self.columns if self._get_computing_column()])

    def get_shape(self) -> tuple[int, ...]:
        col_shape_total = np.sum([col.get_internal_shape()[1] for col in self._get_computing_column()])
        # We assume get_shape picks the first column shape as the row shape
        row_shape_total = self.columns[0].get_internal_shape()[0]
        return row_shape_total, col_shape_total

    def _get_computing_column(self):
        return [col for col in self.columns if isinstance(col, ComputingColumn)]

    def _self_pk_integrity(self):
        return self.get_primary_key().contains_unique()

    def preprocess(self) -> "Table":
        new_cols = self.processor.process(self.columns)
        return Table(new_cols, self.processor, self.pk_col_index)

    def postprocess(self) -> "Table":
        new_cols = self.processor.inverse_process(self.columns)
        return Table(new_cols, self.processor, self.pk_col_index)
