from sdg_core_lib.post_process.functions.Parameter import Parameter
from sdg_core_lib.post_process.functions.UnspecializedFunction import (
    UnspecializedFunction,
    Priority,
)

from abc import ABC


class MonoThreshold(UnspecializedFunction, ABC):
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
        allowed_parameters = [param.name for param in type(self).parameters]
        param_mapping = {
            param.name: param
            for param in self.parameters
            if param.name in allowed_parameters
        }
        for name, param in param_mapping.items():
            setattr(self, name, param.value)
