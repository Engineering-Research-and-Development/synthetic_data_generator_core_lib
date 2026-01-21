import numpy as np
from scipy.stats import normaltest, ttest_1samp, kstest, norm

from sdg_core_lib.post_process.functions.UnspecializedFunction import (
    UnspecializedFunction,
    Priority,
)
from sdg_core_lib.post_process.functions.Parameter import Parameter


class NormalTester(UnspecializedFunction):
    parameters = [
        Parameter("mean", "0.0", "float"),
        Parameter("standard_deviation", "1.0", "float"),
    ]
    description = "Checks if data is normally distributed given a desired mean and standard deviation"
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

    def apply(
        self, n_rows: int, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Checks if data is normally distributed.
        Consider the null hypotesis that data is normally distributed.
        If null hypotesis is rejected (p < 0.05), it means that data is not normally distributed
        Evaluation is based on 3 tests:
        1. D’Agostino and Pearson’s test
        2. Student's t-test
        3. Kolmogorov-Smirnov

        :param data:
        :param n_rows:
        :return: False if null hypotesis is rejected (p < 0.05), True if it is failed to reject (p > 0.05)
        """

        def cdf_function(x):
            return norm.cdf(x, loc=self.mean, scale=self.standard_deviation)

        _, p_normal = normaltest(data)
        _, p_t = ttest_1samp(data, self.mean)
        _, p_k = kstest(data, cdf_function)
        p = min(p_normal, p_t, p_k)

        return np.empty((n_rows, 1)), np.empty((n_rows, 1)), p > 0.05
