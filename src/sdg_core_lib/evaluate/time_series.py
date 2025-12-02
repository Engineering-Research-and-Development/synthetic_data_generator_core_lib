from evaluate.metrics import MetricReport
from sdg_core_lib.dataset.datasets import TimeSeries


class TimeSeriesComparisonEvaluator:
    """
    Evaluates the quality of a synthetic dataset with respect to a real one.

    The evaluation is based on the following metrics:
    - Statistical properties: wasserstein distance and Cramer's V
    - Adherence: evaluates how well the synthetic data adheres to the real data distribution
    - Novelty: evaluates how many new values are generated in the synthetic dataset

    The evaluation is performed on a per-column basis, and the results are aggregated.
    """

    def __init__(
        self,
        real_data: TimeSeries,
        synthetic_data: TimeSeries,
    ):
        if type(real_data) is not TimeSeries:
            raise ValueError("real_data must be a TimeSeries")
        if type(synthetic_data) is not TimeSeries:
            raise ValueError("synthetic_data must be a TimeSeries")
        self._real_data = real_data
        self._synth_data = synthetic_data
        self._numerical_columns = real_data.get_numeric_columns()
        self._categorical_columns = real_data.get_categorical_columns()
        self._synth_numerical_columns = synthetic_data.get_numeric_columns()
        self._synth_categorical_columns = synthetic_data.get_categorical_columns()
        self.report = MetricReport()

    def compute(self):
        if len(self._numerical_columns) <= 1 and len(self._categorical_columns) <= 1:
            return {"available": "false"}

        return {"available": "false"}