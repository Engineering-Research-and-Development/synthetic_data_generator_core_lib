from sdg_core_lib.types import ColumnType
from sdg_core_lib.dataset.Column import ColumnMetadata
from sdg_core_lib.dataset.TypedSubDataset import TypedSubDataset
from sdg_core_lib.evaluate.metrics.base import Metric, MetricType, ComputeStrategy

import numpy as np
import pandas as pd


class CategoricalCramerVStrategy(ComputeStrategy):
    @staticmethod
    def _compute_cramer_v(data1: np.ndarray, data2: np.ndarray):
        """
        Computes Cramer's V on a pair of categorical columns
        :param data1: first column
        :param data2: second column
        :return: Cramer's V
        """
        import scipy.stats as ss
        #import pandas as pd

        confusion_matrix = pd.crosstab(data1, data2)
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        # Total number of observations.
        n = confusion_matrix.to_numpy().sum()
        if n == 0:
            return 0.0
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        # Check for potential division by zero in the correction terms.
        if n - 1 == 0:
            return 0.0
        phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        r_corr = r - ((r - 1) ** 2) / (n - 1)
        k_corr = k - ((k - 1) ** 2) / (n - 1)
        denominator = min(k_corr - 1, r_corr - 1)
        if denominator <= 0:
            return 0.0
        v = np.sqrt(phi2_corr / denominator)
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


if __name__ == "__main__":
    real_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    synthetic_data = pd.DataFrame({"A": [1, 2, -3], "B": [-3, -4, 5]})

    real_data_metadata = [ColumnMetadata("A", position=0, column_type="categorical", data_type="int32"), ColumnMetadata("B", position=1, column_type="categorical", data_type="int32")]
    synthetic_data_metadata = [ColumnMetadata("A", position=0, column_type="categorical", data_type="int32"), ColumnMetadata("B", position=1, column_type="categorical", data_type="int32")]
    real_data = TypedSubDataset.from_data_and_metadata(real_data.to_numpy(), real_data_metadata)
    synthetic_data = TypedSubDataset.from_data_and_metadata(synthetic_data.to_numpy(), synthetic_data_metadata)

    cramer_v = CramerV()
    cramer_v.evaluate(real_data, synthetic_data)
    print(cramer_v.to_json())