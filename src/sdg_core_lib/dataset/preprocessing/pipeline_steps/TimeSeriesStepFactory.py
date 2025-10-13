from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sdg_core_lib.dataset.preprocessing.PipelineStepFactory import PipelineStepFactory
from sdg_core_lib.dataset.preprocessing.pipeline_steps.scalers.TimeSeriesScaler import TimeSeriesScaler


class TimeSeriesStepFactory(PipelineStepFactory):

    @staticmethod
    def create_scaler(scaler: MinMaxScaler | StandardScaler) -> TimeSeriesScaler:
        return TimeSeriesScaler(scaler)