from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray

from sdg_core_lib.dataset.preprocessing.PreProcessingPipeline import (
    PreProcessingPipeline,
)


class Preprocessor(ABC):
    @staticmethod
    @abstractmethod
    def create_preprocessing_pipeline(pipeline: str) -> PreProcessingPipeline:
        raise NotImplementedError


    def execute_preprocessing_pipeline(
        self, pipeline_type: str, data: np.ndarray, test_data: np.ndarray
    ) -> tuple[PreProcessingPipeline, ndarray, ndarray]:
        pipeline = self.create_preprocessing_pipeline(pipeline_type)

        train_data_preprocessed, test_data_preprocessed = pipeline.compute(train_data=data, test_data=test_data)
        return pipeline, train_data_preprocessed, test_data_preprocessed
