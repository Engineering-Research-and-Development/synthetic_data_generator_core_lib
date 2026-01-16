import numpy as np

from sdg_core_lib.post_process.functions.UnspecializedFunction import (
    UnspecializedFunction,
    Priority,
)
from sdg_core_lib.post_process.functions.Parameter import Parameter


class NormalDistributionSample(UnspecializedFunction):
    parameters = [
        Parameter("mean", "0.0", "float"),
        Parameter("standard_deviation", "1.0", "float"),
    ]
    description = "Generates random data from a gaussian distribution with custom mean and standard deviation"
    priority = Priority.MINIMAL
    is_generative = False

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
            raise ValueError("Standard Deviation cannot be less than 0")

    def apply(self, n_rows: int, data: np.ndarray) -> np.ndarray:
        """
        Creates a straight line, sampling n_rows data points from a line y=mx+q

        :param data:
        :param n_rows:
        """
        data = np.random.normal(
            loc=self.mean, scale=self.standard_deviation, size=n_rows
        )
        return data.reshape(-1, 1)
