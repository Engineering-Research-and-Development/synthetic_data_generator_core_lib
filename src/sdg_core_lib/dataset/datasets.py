from abc import ABC, abstractmethod
import numpy as np

from sdg_core_lib.dataset.columns import Numeric, Categorical, Column
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
    def get_computing_data(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_computing_shape(self) -> tuple[int, ...]:
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
        "continuous": Numeric,
        "categorical": Categorical,
        "primary_key": Column,
        "group_index": Column,
    }
    processor: TableProcessor

    def __init__(self, columns: list[Column], processor: TableProcessor, pk_indexes: list[int]= None):
        super().__init__(processor)
        self.columns = columns
        self.pk_col_indexes = pk_indexes
        self.shape = self.get_computing_shape()


    @classmethod
    def from_json(cls, json_data: list[dict], processor: TableProcessor) -> 'Table':
        pk_indexes = []
        columns = []
        for idx, col_data in enumerate(json_data):
            col_type = col_data.get("column_type", "")
            col_name = col_data.get("column_name", "")
            col_values = np.array(col_data.get("column_data", [])).reshape(-1, 1)
            col_value_type = col_data.get("column_datatype", "")
            col_position = idx

            if col_type == "group_index":
                raise NotImplementedError("Tables don't support data grouping. Please, refer to TimeSeries Documentation")

            if col_type == "primary_key":
                pk_indexes.append(col_position)

            col = cls.col_registry.get(col_type, None)(
                col_name,
                col_value_type,
                col_position,
                col_values,
                col_type
            )
            columns.append(col)

        return Table(columns, processor, pk_indexes)

    def clone(self, data: np.ndarray) -> 'Table':
        if self.get_computing_shape()[-1] != data.shape[-1]:
            raise ValueError("Data does not match table shape on column axis")
        n_rows = data.shape[0]
        new_columns = []
        data_idx = 0
        for col in self.columns:
            if col.position not in self.pk_col_indexes:
                # Pick current internal column shape
                col_shape = col.get_internal_shape()[1]
                # Insert data following the correct shape
                data_to_insert = data[:, data_idx:data_idx+col_shape]
                # update data index
                data_idx += col_shape
            else:
                # TODO: Improve primary key generation with a method
                data_to_insert = np.arange(n_rows).reshape(-1, 1)

            new_columns.append(
                type(col)(
                    col.name,
                    col.value_type,
                    col.position,
                    data_to_insert,
                    col.column_type
                )
            )

        return Table(new_columns, self.processor, self.pk_col_indexes)

    def to_json(self) -> list[dict]:
        return [
            {
                "column_data": col.values.reshape(-1,).tolist(),
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

    def get_primary_keys(self) -> list[Column]:
        return [col for col in self.columns if col.position in self.pk_col_indexes]

    def get_computing_data(self) -> np.ndarray:
        return np.hstack([col.get_data() for col in self._get_computing_column()])

    def get_computing_shape(self) -> tuple[int, ...]:
        return self.get_computing_data().shape

    def _get_computing_column(self):
        return [col for col in self.columns if col.position not in self.pk_col_indexes]

    def _self_pk_integrity(self) -> bool:
        pks_values = np.hstack([col.get_data() for col in self.get_primary_keys()])
        return pks_values.shape[0] == np.unique(pks_values, axis=0).shape[0]

    def preprocess(self) -> "Table":
        new_cols = self.processor.process(self.columns)
        return Table(new_cols, self.processor, self.pk_col_indexes)

    def postprocess(self) -> "Table":
        new_cols = self.processor.inverse_process(self.columns)
        return Table(new_cols, self.processor, self.pk_col_indexes)


class TimeSeries(Table):
    def __init__(self, columns: list[Column], processor: TableProcessor, pk_indexes: list[int]= None, group_index: int=None):
        self.group_index = group_index
        super().__init__(columns, processor, pk_indexes)


    @classmethod
    def from_json(cls, json_data: list[dict], processor: TableProcessor) -> 'TimeSeries':
        pk_indexes = []
        group_index = None
        columns = []
        for idx, col_data in enumerate(json_data):
            col_type = col_data.get("column_type", "")
            col_name = col_data.get("column_name", "")
            col_values = np.array(col_data.get("column_data", [])).reshape(-1, 1)
            col_value_type = col_data.get("column_datatype", "")
            col_position = idx

            if col_type == "group_index":
                if group_index is not None:
                    raise ValueError("Group index already set")
                group_index = col_position
                pk_indexes.append(col_position)


            if col_type == "primary_key":
                pk_indexes.append(col_position)

            col = cls.col_registry.get(col_type, None)(
                col_name,
                col_value_type,
                col_position,
                col_values,
                col_type
            )
            columns.append(col)

        if group_index is None:
            raise ValueError("Time series must have a group index to identify isolated experiments")

        return TimeSeries(columns, processor, pk_indexes, group_index)


    def clone(self, data: np.ndarray) -> 'TimeSeries':
        if len(data.shape) == 3:
            # collapsing time steps
            time_steps = data.shape[1]
            data = data.reshape(-1, data.shape[2])
        else:
            raise ValueError("Data must be a 3D array")
        sub_table = super().clone(data)
        # Generate new Group Index and update pk_column
        new_group_index = np.repeat(np.arange(data.shape[0]/time_steps, dtype="int"), repeats=time_steps).reshape(-1, 1)
        new_group_col = type(sub_table.columns[self.group_index])(
            sub_table.columns[self.group_index].name,
            sub_table.columns[self.group_index].value_type,
            sub_table.columns[self.group_index].position,
            new_group_index,
            sub_table.columns[self.group_index].column_type,
        )
        sub_table.columns[self.group_index] = new_group_col
        return TimeSeries(sub_table.columns, sub_table.processor, sub_table.pk_col_indexes, self.group_index)


    def _get_experiment_length(self):
        # TODO: Currently we support only same-shape experiments
        exp_len = None
        experiment_column = self.columns[self.group_index]
        experiment_indexes = experiment_column.get_data().reshape(-1,)
        experiment_numbers = np.unique(experiment_indexes, axis=0)

        for number in experiment_numbers.reshape(-1,):
            experiment_row = np.argwhere(experiment_indexes == number)
            if exp_len is None:
                exp_len = experiment_row.shape[0]
            elif exp_len != experiment_row.shape[0]:
                raise ValueError("Experiments have different lengths")

        return exp_len

    def get_computing_data(self) -> np.ndarray:
        time_steps = self._get_experiment_length()
        data = np.hstack([col.get_data() for col in self._get_computing_column()])
        return data.reshape(-1, time_steps, data.shape[1])

    def preprocess(self) -> "TimeSeries":
        new_cols = self.processor.process(self.columns)
        return TimeSeries(new_cols, self.processor, self.pk_col_indexes, self.group_index)

    def postprocess(self) -> "TimeSeries":
        new_cols = self.processor.inverse_process(self.columns)
        return TimeSeries(new_cols, self.processor, self.pk_col_indexes, self.group_index)





