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
        self.std = None
        super().__init__(parameters)

    def _check_parameters(self):
        param_mapping = {param.name: param for param in self.parameters}
        self.mean = param_mapping["mean"].value
        self.std = param_mapping["standard_deviation"].value

    def _compute(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        noise = np.random.normal(self.mean, self.std, data.shape)
        return data + noise, np.array(range(len(data)))

    def _evaluate(self, data: np.ndarray) -> bool:
        return True
