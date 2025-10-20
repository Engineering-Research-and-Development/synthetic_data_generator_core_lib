from sdg_core_lib.data_type import DataType
from sdg_core_lib.processing.processors.NumericProcessor import NumericProcessor
from sdg_core_lib.processing.processors.TimeSeriesProcessor import TimeSeriesProcessor

processor_map = {
    DataType.NUMERIC: NumericProcessor,
    DataType.TIMESERIES: TimeSeriesProcessor,
}
