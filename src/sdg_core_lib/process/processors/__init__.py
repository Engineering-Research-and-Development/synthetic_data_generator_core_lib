from sdg_core_lib.types import ColumnType
from sdg_core_lib.process.processors.NumericProcessor import NumericProcessor
from sdg_core_lib.process.processors.TimeSeriesProcessor import TimeSeriesProcessor


class ProcessorRegistry:
    @staticmethod
    def get_processor(data_type: ColumnType):
        if data_type == ColumnType.NUMERIC:
            return NumericProcessor()
        elif data_type == ColumnType.TIMESERIES:
            return TimeSeriesProcessor()
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
