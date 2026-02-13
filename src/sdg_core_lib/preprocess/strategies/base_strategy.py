from sdg_core_lib.dataset.columns import Column
from sdg_core_lib.preprocess.strategies.steps import Step
from loguru import logger


class BasePreprocessingStrategy:
    @staticmethod
    def get_steps_per_feature(feature: Column) -> list[Step]:
        logger.warning(
            "You are processing a feature with the base strategy. This will lead to an empty processing pipeline"
        )
        return []
