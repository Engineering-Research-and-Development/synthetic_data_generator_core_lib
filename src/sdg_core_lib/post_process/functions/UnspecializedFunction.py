import numpy as np
from abc import ABC, abstractmethod
from enum import Enum

from sdg_core_lib.post_process.functions.Parameter import Parameter


class Priority(Enum):
    MINIMAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    MAX = 5
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
    def apply(
        self, n_rows: int, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Function application interface: class implementation defines how to apply the function to the data
        :param n_rows: how many rows have to be affected (optional for some functions)
        :param data: data on which function have to be applied (optional for some functions)
        :return:
            - new data: computational results
            - indexes: indexes of the rows that have been affected
            - success_flag: boolean flag indicating if the function has been applied correctly.
                For statistical tests, it is False if the test fails
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
