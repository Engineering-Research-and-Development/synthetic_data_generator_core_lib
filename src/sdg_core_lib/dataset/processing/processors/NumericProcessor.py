from sdg_core_lib.dataset.processing.processors.PipelineConfig import (
    PipelineConfig,
)
from sdg_core_lib.dataset.processing.processors.ProcessingPipeline import (
    ProcessingPipeline,
)
from sdg_core_lib.dataset.processing.Processor import Processor
from sdg_core_lib.dataset.processing.processors.pipeline_steps.NumericStepFactory import (
    NumericStepFactory,
)


class NumericProcessor(Processor):
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.pipeline = self.create_processing_pipeline()

    def create_processing_pipeline(self) -> ProcessingPipeline:
        step_factory = NumericStepFactory()
        pipeline = ProcessingPipeline(step_factory, self.config)
        return pipeline
