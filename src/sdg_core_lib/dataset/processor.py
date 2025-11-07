from abc import ABC, abstractmethod
from sdg_core_lib.dataset.steps import Step
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
    def process(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def inverse_process(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TableProcessor(Processor):
    def __init__(self, dir_path: str):
        super().__init__(dir_path)