from sdg_core_lib.dataset.preprocessing.PreProcessingPipeline import PreProcessingPipeline
from sdg_core_lib.dataset.preprocessing.Preprocessor import Preprocessor
from sdg_core_lib.dataset.preprocessing.pipeline_steps.TimeSeriesStepFactory import TimeSeriesStepFactory
from sdg_core_lib.dataset.preprocessing.pipelines.ScalePipeline import ScalePipeline


class TimeSeriesPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()

    def create_preprocessing_pipeline(self, pipeline: str) -> PreProcessingPipeline:
        step_factory = TimeSeriesStepFactory()

        if pipeline == "scale":
            return ScalePipeline(step_factory)
        else:
            raise NotImplementedError