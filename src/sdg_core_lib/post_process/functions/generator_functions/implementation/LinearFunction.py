import numpy as np

from sdg_core_lib.post_process.functions.UnspecializedFunction import (
    UnspecializedFunction,
    Priority,
)
from sdg_core_lib.post_process.functions.Parameter import Parameter
from sdg_core_lib.post_process.function_utils import check_min_max_boundary


class LinearFunction(UnspecializedFunction):
    parameters = [
        Parameter("m", "1.0", "float"),
        Parameter("q", "0.0", "float"),
        Parameter("min_value", "0.0", "float"),
        Parameter("max_value", "1.0", "float"),
    ]
    description = "Generates linear data in domain comprised between min_value and max_value following the y=mx+q equation"
    priority = Priority.MINIMAL
    is_generative = False

    def __init__(self, parameters: list[Parameter]):
        self.m = None
        self.q = None
        self.min_value = None
        self.max_value = None
        super().__init__(parameters)

    def _check_parameters(self):
        param_mapping = {param.name: param for param in self.parameters}
        self.m = param_mapping["m"].value
        self.q = param_mapping["q"].value
        self.min_value = param_mapping["min_value"].value
        self.max_value = param_mapping["max_value"].value
        check_min_max_boundary(self.min_value, self.max_value)

    def apply(self, n_rows: int, data: np.ndarray) -> bool:
        """
        Creates a straight line, sampling n_rows data points from a line y=mx+q

        :param data:
        :param n_rows:
        """
        data = np.linspace(self.min_value, self.max_value, n_rows)
        data = self.m * data + self.q

        return data
