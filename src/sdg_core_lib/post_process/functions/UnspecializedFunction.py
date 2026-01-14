import numpy as np
from abc import ABC, abstractmethod
from enum import Enum

from sdg_core_lib.post_process.functions.Parameter import Parameter


class Priority(Enum):
    MINIMAL = 5
    LOW = 4
    MEDIUM = 3
    HIGH = 2
    MAX = 1
    NONE = None


class UnspecializedFunction(ABC):
    parameters: list[Parameter] = (None,)
    description: str = None
    priority: Priority = Priority.NONE
    is_generative: bool = None

    def __init__(self, parameters: list[Parameter]):
        self.parameters = parameters
        self._check_parameters()

    @classmethod
    def from_json(cls, json_params):
        return cls([Parameter.from_json(param) for param in json_params])

    @abstractmethod
    def _check_parameters(self):
        raise NotImplementedError

    @abstractmethod
    def _compute(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Applies a data transformation function on a given set of generated data
        :param data: a numpy array of data from a single feature
        :return: transformed data and affected indexes
        """
        raise NotImplementedError

    @abstractmethod
    def _evaluate(self, data: np.ndarray) -> bool:
        """
        Applies an evaluation function on a given set of generated data
        :param data: a numpy array of data from a single feature
        :return: a single boolean value evaluating id data meets evaluation criteria
        """
        raise NotImplementedError

    @classmethod
    def self_describe(cls):
        return {
            "function": {
                "name": f"{cls.__qualname__}",
                "description": cls.description,
                "function_reference": f"{cls.__module__}.{cls.__qualname__}",
                "priority": cls.priority.value,
                "is_generative": cls.is_generative,
            },
            "parameters": [param.to_json() for param in cls.parameters],
        }
