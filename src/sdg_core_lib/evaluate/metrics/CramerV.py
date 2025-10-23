from sdg_core_lib.types import ColumnType
from sdg_core_lib.dataset.Column import ColumnMetadata
from sdg_core_lib.dataset.TypedSubDataset import TypedSubDataset
from sdg_core_lib.evaluate.metrics.base import Metric, MetricType, ComputeStrategy

import numpy as np


class CategoricalCramerVStrategy(ComputeStrategy):
    @staticmethod
    def _compute_cramer_v(data1: np.ndarray, data2: np.ndarray):
        """
        Computes Cramer's V on a pair of categorical columns.
        Careful: Cramer's V is biased if there are missing categories!
        :param data1: first column
        :param data2: second column
        :return: Cramer's V
        """
        import scipy.stats as ss
        import pandas as pd

        observed = pd.crosstab(data1, data2)
        v = ss.contingency.association(observed)
        return v

    @staticmethod
    def compute(real_data: TypedSubDataset, synthetic_data: TypedSubDataset):
        contingency_scores_distances = {}
        for idx in range(len(real_data.columns)):
            for idx_2 in range(len(real_data.columns))[idx + 1 :]:
                real_col = real_data.columns[idx]
                real_col_2 = real_data.columns[idx_2]
                synthetic_col = synthetic_data.columns[idx]
                synthetic_col_2 = synthetic_data.columns[idx_2]
                v_real = CategoricalCramerVStrategy._compute_cramer_v(
                    real_col.data, real_col_2.data
                )
                v_synth = CategoricalCramerVStrategy._compute_cramer_v(
                    synthetic_col.data, synthetic_col_2.data
                )
                contingency_scores_distances[
                    f"{real_col.metadata.name} over {real_col_2.metadata.name}"
                ] = (
                    np.round(np.clip(1 - (np.abs(v_real - v_synth)), 0, 1), 2).item()
                    * 100
                )

        return contingency_scores_distances


class CramerV(Metric):
    def __init__(self):
        super().__init__(
            title="Cramer's V",
            unit_measure="%",
            metric_type=MetricType.STATISTICAL,
            min_cols=2,
        )
        self._strategies.register(ColumnType.CATEGORICAL, CategoricalCramerVStrategy())

