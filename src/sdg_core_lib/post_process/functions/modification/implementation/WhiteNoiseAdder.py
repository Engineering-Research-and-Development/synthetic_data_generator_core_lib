from sdg_core_lib.post_process.functions.Parameter import Parameter
from sdg_core_lib.post_process.functions.UnspecializedFunction import (
    UnspecializedFunction,
    Priority,
)
import numpy as np


class WhiteNoiseAdder(UnspecializedFunction):
    parameters = [
        Parameter("mean", "0.0", "float"),
        Parameter("standard_deviation", "1.0", "float"),
    ]
    description = "Adds white noise to the data"
    is_generative = False
    priority = Priority.LOW

    def __init__(self, parameters: list[Parameter]):
        self.mean = None
        self.standard_deviation = None
        super().__init__(parameters)

    def _check_parameters(self):
        allowed_parameters = [param.name for param in type(self).parameters]
        param_mapping = {
            param.name: param
            for param in self.parameters
            if param.name in allowed_parameters
        }
        for name, param in param_mapping.items():
            setattr(self, name, param.value)
        if self.standard_deviation < 0:
            raise ValueError("standard_deviation cannot be less than 0")

    def apply(self, n_rows: int, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        noise = np.random.normal(self.mean, self.standard_deviation, data.shape)
        return data + noise, np.array(range(len(data)))
