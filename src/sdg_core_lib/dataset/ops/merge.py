from sdg_core_lib.dataset.DatasetComponent import DatasetComponent
from sdg_core_lib.dataset.Dataset import Dataset


def merge_all_datasets(subdatasets: list[DatasetComponent]) -> Dataset:
    columns = []
    for subdataset in subdatasets:
        columns.extend(subdataset.columns)
    return Dataset(columns)
