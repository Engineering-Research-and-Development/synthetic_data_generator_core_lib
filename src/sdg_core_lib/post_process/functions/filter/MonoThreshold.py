from sdg_core_lib.post_process.functions.Parameter import Parameter
from sdg_core_lib.post_process.functions.UnspecializedFunction import (
    UnspecializedFunction,
    Priority,
)

import numpy as np


class MonoThreshold(UnspecializedFunction):
    parameters = [
        Parameter("value", "0.0", "float"),
        Parameter("strict", "True", "bool"),
    ]
    priority = Priority.MINIMAL
    is_generative = False

    def __init__(self, parameters: list[Parameter]):
        self.value = None
        self.strict = None
        super().__init__(parameters)

    def _check_parameters(self):
        param_mapping = {param.name: param for param in self.parameters}
        self.value = param_mapping["value"].value
        self.strict = param_mapping["strict"].value

    def _compute(self, data: np.ndarray):
        pass

    def _evaluate(self, data: np.ndarray):
        pass
