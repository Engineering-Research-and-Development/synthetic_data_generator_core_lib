import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sdg_core_lib.dataset.preprocessing.PipelineStepFactory import PipelineStepFactory
from sdg_core_lib.dataset.preprocessing.PreProcessingPipeline import PreProcessingPipeline


class ScalePipeline(PreProcessingPipeline):
    def __init__(self, step_factory: PipelineStepFactory):
        super().__init__()
        self.step_factory = step_factory

    def _prepare(self, scaler: MinMaxScaler | StandardScaler):
        self.scaler = self.step_factory.create_scaler(scaler)

    def compute(self, train_data: np.ndarray, test_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.scaler.is_fit:
            return self.scaler.transform(train_data, test_data)
        else:
            return self.scaler.fit_transform(train_data, test_data)