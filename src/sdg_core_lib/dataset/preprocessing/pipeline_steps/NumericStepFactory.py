from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sdg_core_lib.dataset.preprocessing.PipelineStepFactory import PipelineStepFactory
from sdg_core_lib.dataset.preprocessing.pipeline_steps.scalers.NumericScaler import NumericScaler


class NumericStepFactory(PipelineStepFactory):

    @staticmethod
    def create_scaler(scaler: MinMaxScaler | StandardScaler) -> NumericScaler:
        return NumericScaler(scaler)