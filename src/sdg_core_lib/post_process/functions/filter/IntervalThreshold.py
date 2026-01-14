from sdg_core_lib.post_process.functions.UnspecializedFunction import (
    UnspecializedFunction, Priority
)
from sdg_core_lib.post_process.functions.Parameter import Parameter
import numpy as np

class IntervalThreshold(UnspecializedFunction):
    parameters = [
        Parameter("lower_bound", "0.0", "float"),
        Parameter("upper_bound", "1.0", "float"),
        Parameter("lower_strict", "True", "bool"),
        Parameter("upper_strict", "True", "bool"),
    ]
    priority = Priority.MINIMAL
    is_generative = False

    def __init__(self, parameters: list[Parameter]):
        self.upper_bound = None
        self.lower_bound = None
        self.upper_strict = None
        self.lower_strict = None
        super().__init__(parameters)

    def _check_parameters(self):
        param_mapping = {param.name: param for param in self.parameters}
        self.upper_bound = param_mapping["upper_bound"].value
        self.lower_bound = param_mapping["lower_bound"].value
        self.upper_strict = param_mapping["upper_strict"].value
        self.lower_strict = param_mapping["lower_strict"].value

    def _compute(self, data: np.ndarray):
        pass

    def _evaluate(self, data: np.ndarray):
        pass
