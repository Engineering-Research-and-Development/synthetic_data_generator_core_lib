from abc import ABC, abstractmethod

from dataset import ColumnRegistry
from dataset.steps import OneHotEncoderWrapper
from sdg_core_lib.dataset import Column, NumericColumn, CategoricalColumn
from sdg_core_lib.dataset.steps import Step, ScalerWrapper
import numpy as np


class Processor(ABC):
    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        self.steps: list[Step] = []

    def add_step(self, step: Step) -> 'Processor':
        self.steps.append(step)
        return self

    def _save_all(self):
        [step.save(self.dir_path) for step in self.steps]

    def _load_all(self) -> 'Processor':
        [step.load(self.dir_path) for step in self.steps]
        return self

    @abstractmethod
    def process(self, *args) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def inverse_process(self, *args) -> np.ndarray:
        raise NotImplementedError


class TableProcessor(Processor):
    def __init__(self, dir_path: str):
        super().__init__(dir_path)

    # TODO: 'sta cosa non mi piace. Scendo a compromessi per uno scaler per colonna,
    #  ma la modalità la deve decidere qualcuno e non devo impazzire a creare
    #  costruttori enormi per il processor
    def _init_steps(self, columns: list[Column]):
        for col in columns:
            if isinstance(col, NumericColumn):
                self.add_step(ScalerWrapper(col.position, "standard"))
            elif isinstance(col, CategoricalColumn):
                self.add_step(OneHotEncoderWrapper(col.position, "onehot"))

    def process(self, columns: list[Column]) -> list[Column]:
        preprocessed_columns = []
        self._init_steps(columns)
        for idx, col in enumerate(columns):
            result = self.steps[idx].fit_transform(col.get_data())
            preprocessed_columns.append(
                ColumnRegistry.get_column(col.column_type)(
                    col.name,
                    col.value_type,
                    col.position,
                    result,
                )
            )
        self._save_all()
        return preprocessed_columns


    def inverse_process(self, preprocessed_columns: list[Column]) -> list[Column]:
        self._init_steps(preprocessed_columns)
        self._load_all()
        post_processed_columns = []
        for idx, col in enumerate(preprocessed_columns):
            result = self.steps[idx].inverse_transform(col.get_data())
            post_processed_columns.append(
                ColumnRegistry.get_column(col.column_type)(
                    col.name,
                    col.value_type,
                    col.position,
                    result,
                )
            )
        return post_processed_columns
