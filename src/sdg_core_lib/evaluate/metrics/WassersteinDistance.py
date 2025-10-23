from sdg_core_lib.types import ColumnType
from sdg_core_lib.dataset.TypedSubDataset import TypedSubDataset
from sdg_core_lib.evaluate.metrics.base import Metric, MetricType, ComputeStrategy

import numpy as np


class NumericWassersteinDistanceStrategy(ComputeStrategy):
    @staticmethod
    def compute(real_data: TypedSubDataset, synthetic_data: TypedSubDataset):
        import scipy.stats as ss

        wass_distance_scores = {}
        for real_col, synthetic_col in zip(real_data.columns, synthetic_data.columns):
            real_data = real_col.data
            synth_data = synthetic_col.data
            print(real_data.shape)
            distance = np.abs(np.max(real_data) - np.min(real_data))
            wass_dist = ss.wasserstein_distance(real_data, synth_data)
            wass_dist = np.clip(wass_dist, 0, distance) / distance
            wass_distance_scores[real_col.metadata.name] = (
                np.round(1 - wass_dist, 2).item() * 100
            )

        return wass_distance_scores


class WassersteinDistance(Metric):
    def __init__(self):
        super().__init__(
            title="Wasserstein Distance",
            unit_measure="%",
            metric_type=MetricType.STATISTICAL,
        )
        self._strategies.register(
            ColumnType.NUMERIC, NumericWassersteinDistanceStrategy()
        )
