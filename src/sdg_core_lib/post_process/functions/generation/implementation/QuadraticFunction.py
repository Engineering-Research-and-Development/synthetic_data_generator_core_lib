import numpy as np

from sdg_core_lib.post_process.functions.UnspecializedFunction import (
    UnspecializedFunction,
    Priority,
)
from sdg_core_lib.post_process.functions.Parameter import Parameter
from sdg_core_lib.post_process.function_utils import check_min_max_boundary


class QuadraticFunction(UnspecializedFunction):
    parameters = [
        Parameter("a", "1.0", "float"),
        Parameter("b", "0.0", "float"),
        Parameter("c", "0.0", "float"),
        Parameter("min_value", "0.0", "float"),
        Parameter("max_value", "1.0", "float"),
    ]
    description = "Generates parabolic data in domain comprised between min_value and max_value following the y=a^2+bx+c equation"
    priority = Priority.MAX
    is_generative = False

    def __init__(self, parameters: list[Parameter]):
        self.a = None
        self.b = None
        self.c = None
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

    def apply(
        self, n_rows: int, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Creates a parabola, sampling n_rows data points from a line y=ax^2 + bx + c

        :param data:
        :param n_rows:
        """
        data = np.linspace(self.min_value, self.max_value, n_rows)
        data = self.a * data**2 + self.b * data + self.c

        return data.reshape(-1, 1), np.empty((n_rows, 1)), True
