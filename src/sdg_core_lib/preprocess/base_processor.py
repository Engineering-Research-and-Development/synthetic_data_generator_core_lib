from abc import ABC, abstractmethod

from sdg_core_lib.preprocess.strategies.base_strategy import BasePreprocessingStrategy
from sdg_core_lib.preprocess.strategies.steps import Step
import numpy as np


class Processor(ABC):
    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        self.steps: dict[int, list[Step]] = {}
        self.idx_to_data: dict[int, int] = {}
        self.strategy: BasePreprocessingStrategy = BasePreprocessingStrategy()

    @abstractmethod
    def _init_steps(self, data: list):
        raise NotImplementedError

    def set_strategy(self, strategy: BasePreprocessingStrategy):
        self.strategy = strategy
        return self

    def add_steps(
        self, steps: list[Step], col_position: int, data_position: int
    ) -> "Processor":
        self.steps[col_position] = steps
        self.idx_to_data[col_position] = data_position
        return self

    def save_all(self):
        [
            step.save_if_not_exist(self.dir_path)
            for step_list in self.steps.values()
            for step in step_list
        ]

    def load_all(self) -> "Processor":
        [
            step.load(self.dir_path)
            for step_list in self.steps.values()
            for step in step_list
        ]
        return self

    def process(self, data: list) -> dict[int, np.ndarray]:
        results = {
            idx: step.fit_transform(data[self.idx_to_data[idx]])
            for idx, step_list in self.steps.items()
            for step in step_list
        }
        self.save_all()
        return results

    def inverse_process(self, data: list) -> dict[int, np.ndarray]:
        self.load_all()
        return {
            idx: step.inverse_transform(data[self.idx_to_data[idx]])
            for idx, step_list in self.steps.items()
            for step in reversed(step_list)
        }
