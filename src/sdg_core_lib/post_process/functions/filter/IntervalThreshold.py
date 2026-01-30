from abc import ABC

from sdg_core_lib.commons import AllowedData, DataType
from sdg_core_lib.post_process.functions.UnspecializedFunction import (
    UnspecializedFunction,
    Priority,
)
from sdg_core_lib.post_process.function_utils import check_min_max_boundary
from sdg_core_lib.post_process.functions.Parameter import Parameter


class IntervalThreshold(UnspecializedFunction, ABC):
    parameters = [
        Parameter("lower_bound", "0.0", "float"),
        Parameter("upper_bound", "1.0", "float"),
        Parameter("lower_strict", "True", "bool"),
        Parameter("upper_strict", "True", "bool"),
    ]
    priority = Priority.MINIMAL
    is_generative = False
    allowed_data = [
        AllowedData(DataType.float32, False),
        AllowedData(DataType.int32, False),
    ]

    def __init__(self, parameters: list[Parameter]):
        self.upper_bound = None
        self.lower_bound = None
        self.upper_strict = None
        self.lower_strict = None
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
        check_min_max_boundary(self.lower_bound, self.upper_bound)
