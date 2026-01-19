import numpy as np

from sdg_core_lib.post_process.functions.UnspecializedFunction import (
    UnspecializedFunction,
    Priority,
)
from sdg_core_lib.post_process.functions.Parameter import Parameter
from sdg_core_lib.post_process.function_utils import check_min_max_boundary


class SinusoidalFunction(UnspecializedFunction):
    parameters = [
        Parameter("a", "1.0", "float"),
        Parameter("f", "1.0", "float"),
        Parameter("phi", "0.0", "float"),
        Parameter("v", "0.0", "float"),
        Parameter("min_value", "0.0", "float"),
        Parameter("max_value", "1.0", "float"),
    ]
    description = "Generates sinusoidal data in domain comprised between min_value and max_value following the y = a*sin(2*pi*f*x + 2*pi*phi) + v equation"
    priority = Priority.MINIMAL
    is_generative = False

    def __init__(self, parameters: list[Parameter]):
        self.a = None
        self.f = None
        self.phi = None
        self.v = None
        self.min_value = None
        self.max_value = None
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
        check_min_max_boundary(self.min_value, self.max_value)

    def apply(self, n_rows: int, data: np.ndarray) -> np.ndarray:
        """
        Generates the curve in the defined interval

        :param data:
        :param n_rows:
        """
        data = np.linspace(self.min_value, self.max_value, n_rows)
        data = (
            self.a * np.sin(2 * np.pi * self.f * data + 2 * np.pi * self.phi) + self.v
        )

        return data.reshape(-1, 1)
