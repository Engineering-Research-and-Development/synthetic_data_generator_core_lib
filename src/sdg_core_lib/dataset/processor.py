from abc import ABC, abstractmethod

from dataset.steps import NoneStep
from sdg_core_lib.dataset.steps import OneHotEncoderWrapper
from sdg_core_lib.dataset.columns import Column, Numeric, Categorical
from sdg_core_lib.dataset.steps import Step, ScalerWrapper
import numpy as np


class Processor(ABC):
    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        self.steps: dict[int, list[Step]] = {}

    @abstractmethod
    def _init_steps(self, data: list):
        raise NotImplementedError

    def add_step(self, step: Step, data_index: int) -> 'Processor':
        self.steps.get(data_index).append(step)
        return self

    def _save_all(self):
        [step.save(self.dir_path) for step_list in self.steps.values() for step in step_list]

    def _load_all(self) -> 'Processor':
        [step.load(self.dir_path) for step_list in self.steps.values() for step in step_list]
        return self

    def process(self, data: list) -> dict[int, np.ndarray]:
        results = {idx: step.fit_transform(data[idx]) for idx, step_list in self.steps.items() for step in step_list}
        self._save_all()
        return results

    def inverse_process(self, data: list) -> dict[int, np.ndarray]:
        self._load_all()
        return {idx: step.inverse_transform(data[idx]) for idx, step_list in self.steps.items() for step in reversed(step_list)}


class TableProcessor(Processor):
    def __init__(self, dir_path: str):
        super().__init__(dir_path)

    # TODO: External config?
    def _init_steps(self, columns: list[Column]):
        for col in columns:
            self.steps[col.position] = []
            if isinstance(col, Numeric):
                self.add_step(ScalerWrapper(col.position, "standard"), col.position)
            elif isinstance(col, Categorical):
                self.add_step(OneHotEncoderWrapper(col.position), col.position)
            else:
                self.add_step(NoneStep(col.position), col.position)

    def process(self, columns: list[Column]) -> list[Column]:
        self._init_steps(columns)
        col_data = [col.get_data() for col in columns]
        results = super().process(col_data)
        preprocessed_columns = []
        for col in columns:
            preprocessed_columns.append(
                type(col)(
                    col.name,
                    col.value_type,
                    col.position,
                    results.get(col.position),
                    col.column_type
                )
            )
        return preprocessed_columns

    def inverse_process(self, preprocessed_columns: list[Column]) -> list[Column]:
        self._init_steps(preprocessed_columns)
        col_data = [col.get_data() for col in preprocessed_columns]
        results = super().inverse_process(col_data)
        post_processed_columns = []
        for col in preprocessed_columns:
            post_processed_columns.append(
                type(col)(
                    col.name,
                    col.value_type,
                    col.position,
                    results.get(col.position),
                    col.column_type
                )
            )
        return post_processed_columns
