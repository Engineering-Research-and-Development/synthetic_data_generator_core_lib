from sdg_core_lib.dataset.preprocessing.PreProcessingPipeline import PreProcessingPipeline
from sdg_core_lib.dataset.preprocessing.Preprocessor import Preprocessor
from sdg_core_lib.dataset.preprocessing.pipeline_steps.NumericStepFactory import NumericStepFactory
from sdg_core_lib.dataset.preprocessing.pipelines.ScalePipeline import ScalePipeline


class NumericPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()

    def create_preprocessing_pipeline(self, pipeline: str) -> PreProcessingPipeline:
        step_factory = NumericStepFactory()

        if pipeline == "scale":
            return ScalePipeline(step_factory)
        else:
            raise NotImplementedError
