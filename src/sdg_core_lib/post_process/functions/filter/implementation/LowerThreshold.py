import numpy as np

from sdg_core_lib.post_process.functions.Parameter import Parameter
from sdg_core_lib.post_process.functions.filter.MonoThreshold import MonoThreshold


class LowerThreshold(MonoThreshold):
    description = "Mono-threshold function: pick values greater than a lower threshold"

    def __init__(self, parameters: list[Parameter]):
        super().__init__(parameters)

    def apply(self, n_rows: int, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.strict:
            indexes = np.greater_equal(data, self.value)
        else:
            indexes = np.greater(data, self.value)
        return data[indexes], indexes
