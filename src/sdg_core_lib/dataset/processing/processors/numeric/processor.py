from sdg_core_lib.dataset.processing.config.pipeline import (
    PipelineConfig,
)
from sdg_core_lib.dataset.processing.base.pipeline import (
    ProcessingPipeline,
)
from sdg_core_lib.dataset.processing.base.processor import Processor
from sdg_core_lib.dataset.processing.factories.pipeline_steps.numeric import (
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
