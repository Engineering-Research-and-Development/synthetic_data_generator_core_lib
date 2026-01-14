import numpy as np

from sdg_core_lib.post_process.functions.Parameter import Parameter
from sdg_core_lib.post_process.functions.filter.MonoThreshold import MonoThreshold


class UpperThreshold(MonoThreshold):
    description = ("Mono-threshold function: picks value less than an upper threshold",)

    def __init__(self, parameters: list[Parameter]):
        super().__init__(parameters)

    def _compute(self, data: np.ndarray):
        if self.strict:
            indexes = np.less_equal(data, self.value)
        else:
            indexes = np.less(data, self.value)
        return data[indexes], indexes

    def _evaluate(self, data: np.ndarray):
        return True
