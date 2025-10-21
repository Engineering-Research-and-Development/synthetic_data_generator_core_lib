from sdg_core_lib.data_type import DataType
from sdg_core_lib.processing.processors.NumericProcessor import NumericProcessor
from sdg_core_lib.processing.processors.TimeSeriesProcessor import TimeSeriesProcessor


class ProcessorRegistry:
    @staticmethod
    def get_processor(data_type: DataType):
        if data_type == DataType.NUMERIC:
            return NumericProcessor()
        elif data_type == DataType.TIMESERIES:
            return TimeSeriesProcessor()
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
