from typing import Type
from abc import ABC

from sdg_core_lib.data_generator.models.GANs.implementation.CTGAN import CTGAN
from sdg_core_lib.data_generator.models.VAEs.implementation.TabularVAE import (
    TabularVAE,
)
from sdg_core_lib.data_generator.models.VAEs.implementation.TimeSeriesVAE import (
    TimeSeriesVAE,
)
from sdg_core_lib.dataset.datasets import Table, TimeSeries, Dataset
from sdg_core_lib.evaluate.base_evaluator import BaseEvaluator
from sdg_core_lib.preprocess.strategies.base_strategy import BasePreprocessingStrategy
from sdg_core_lib.preprocess.strategies.vae_strategy import (
    TabularVAEPreprocessingStrategy,
    TimeSeriesVAEPreprocessingStrategy,
)
from sdg_core_lib.preprocess.strategies.ctgan_strategy import CTGANPreprocessingStrategy
from sdg_core_lib.evaluate.tables import TabularComparisonEvaluator
from sdg_core_lib.evaluate.time_series import TimeSeriesComparisonEvaluator
from sdg_core_lib.preprocess.table_processor import TableProcessor
from sdg_core_lib.preprocess.base_processor import Processor


class DatasetMapping(ABC):
    mapping = {"dataset": Dataset, "evaluator": BaseEvaluator, "processor": Processor}

    @classmethod
    def get_dataset_class(cls) -> Type[Dataset]:
        """Get the dataset class for a specific dataset type."""
        return cls.mapping["dataset"]

    @classmethod
    def get_evaluator_class(cls) -> Type[BaseEvaluator]:
        """Get the evaluator class for a specific dataset type."""
        return cls.mapping["evaluator"]

    @classmethod
    def get_processor_class(cls) -> Type[Processor]:
        """Get the processor class for a specific dataset type."""
        return cls.mapping["processor"]


class TableMapping(DatasetMapping):
    mapping = {
        "dataset": Table,
        "evaluator": TabularComparisonEvaluator,
        "processor": TableProcessor,
    }


class TimeSeriesMapping(DatasetMapping):
    mapping = {
        "dataset": TimeSeries,
        "evaluator": TimeSeriesComparisonEvaluator,
        "processor": TableProcessor,
    }


class ModelStrategyMapping:
    """Mapping configuration for model types to their preprocessing strategies."""

    mapping = {
        TabularVAE: TabularVAEPreprocessingStrategy,
        TimeSeriesVAE: TimeSeriesVAEPreprocessingStrategy,
        CTGAN: CTGANPreprocessingStrategy,
    }

    @classmethod
    def get_strategy(cls, model_class) -> Type[BasePreprocessingStrategy]:
        """Get the preprocessing strategy for a specific model class."""
        return cls.mapping.get(model_class, BasePreprocessingStrategy)
