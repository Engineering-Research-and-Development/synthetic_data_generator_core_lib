from sdg_core_lib.dataset.columns import Column, Numeric, Categorical
from sdg_core_lib.preprocess.strategies.base_strategy import BasePreprocessingStrategy
from sdg_core_lib.preprocess.strategies.steps import (
    Step,
    ScalerWrapper,
    NoneStep,
    OneHotEncoderWrapper,
)


class TabularVAEPreprocessingStrategy(BasePreprocessingStrategy):
    @staticmethod
    def get_steps_per_feature(feature: Column) -> list[Step]:
        step_list = []
        if isinstance(feature, Numeric):
            step_list.append(ScalerWrapper(feature.position, feature.name, "standard"))
        elif isinstance(feature, Categorical):
            step_list.append(OneHotEncoderWrapper(feature.position, feature.name))
        elif type(feature) is Column:
            step_list.append(NoneStep(feature.position))
        else:
            raise NotImplementedError()
        return step_list


class TimeSeriesVAEPreprocessingStrategy(BasePreprocessingStrategy):
    @staticmethod
    def get_steps_per_feature(feature: Column) -> list[Step]:
        step_list = []
        if isinstance(feature, Numeric):
            step_list.append(ScalerWrapper(feature.position, feature.name, "minmax"))
        elif isinstance(feature, Categorical):
            step_list.append(OneHotEncoderWrapper(feature.position, feature.name))
        elif type(feature) is Column:
            step_list.append(NoneStep(feature.position))
        else:
            raise NotImplementedError()
        return step_list
