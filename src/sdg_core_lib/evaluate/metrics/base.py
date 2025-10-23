from abc import ABC, abstractmethod
from enum import Enum

from sdg_core_lib.types import ColumnType
from sdg_core_lib.dataset.TypedSubDataset import TypedSubDataset


class MetricType(Enum):
    STATISTICAL = "statistical"
    ADHERENCE = "adherence"
    NOVELTY = "novelty"


class ComputeStrategy(ABC):
    @staticmethod
    @abstractmethod
    def compute(real_data: TypedSubDataset, synthetic_data: TypedSubDataset):
        raise NotImplementedError


class StrategyRegistry:
    def __init__(self):
        self.registry = {}

    def register(self, dtype: ColumnType, strategy: ComputeStrategy):
        self.registry[dtype] = strategy

    def get(self, dtype: ColumnType) -> ComputeStrategy | None:
        try:
            return self.registry[dtype]
        except TypeError:
            pass


class Metric(ABC):
    def __init__(
        self, title: str, unit_measure: str, metric_type: MetricType, min_cols: int = 1
    ):
        self.title = title
        self.unit_measure = unit_measure
        self.type = metric_type
        self.min_cols = min_cols
        self._strategies = StrategyRegistry()
        self.value: int | float | dict | None = None

    def to_json(self):
        return {
            "title": self.title,
            "unit_measure": self.unit_measure,
            "value": self.value,
        }

    def _validate(
        self, real_data: TypedSubDataset, synthetic_data: TypedSubDataset
    ) -> None:
        if len(real_data.columns) < self.min_cols:
            raise ValueError(f"Real data must have at least {self.min_cols} columns")
        if len(real_data.columns) != len(synthetic_data.columns):
            raise ValueError(
                f"Real and Synthetic Data should have the same number of columns. Found Real data with {len(real_data.columns)} columns and Synthetic data with {len(synthetic_data.columns)} columns"
            )
        if real_data.get_processing_shape() != synthetic_data.get_processing_shape():
            raise ValueError(
                f"Real and Synthetic Data should have the same shape. Found Real data of shape {real_data.get_processing_shape()} and Synthetic data of shape {synthetic_data.get_processing_shape()}"
            )
        if real_data.data_type != synthetic_data.data_type:
            raise ValueError(
                f"Real and Synthetic Data should be of the same data type. Found Real Data of type {real_data.data_type} and Synthetic Data of type {synthetic_data.data_type}"
            )

    def evaluate(
        self, real_data: TypedSubDataset, synthetic_data: TypedSubDataset
    ) -> None:
        self._validate(real_data, synthetic_data)
        data_type = real_data.data_type
        strategy = self._strategies.get(data_type)
        if strategy is not None:
            self.value = strategy.compute(real_data, synthetic_data)


# TODO: Move to Evaluator
class MetricReport:
    def __init__(self):
        self.report = {}

    def add_metric(self, metric: Metric):
        if metric.type not in self.report:
            self.report[metric.type] = [metric.to_json()]
        else:
            self.report[metric.type].append(metric.to_json())

    def to_json(self):
        if len(self.report) == 0:
            return {}

        return self.report
