from sdg_core_lib.processing.pipeline.ProcessingPipeline import (
    ProcessingPipeline,
)
from sdg_core_lib.processing.processors.Processor import Processor
from sdg_core_lib.processing.factories.NumericStepFactory import (
    NumericStepFactory,
)


class NumericProcessor(Processor):
    def __init__(self):
        super().__init__()

    def create_processing_pipeline(self) -> ProcessingPipeline:
        step_factory = NumericStepFactory()
        pipeline = ProcessingPipeline(step_factory, self.config)
        return pipeline
