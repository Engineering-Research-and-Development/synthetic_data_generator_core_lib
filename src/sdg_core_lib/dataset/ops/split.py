from sdg_core_lib.dataset.Dataset import Dataset
from sdg_core_lib.dataset.TypedSubDataset import TypedSubDataset


def split_into_subdataset(dataset: Dataset) -> list[TypedSubDataset]:
    subdatasets = []
    all_data_types = set([column.metadata.column_type for column in dataset.columns])
    for data_type in all_data_types:
        grouped_columns = [
            col for col in dataset.columns if col.metadata.column_type == data_type
        ]
        subdatasets.append(TypedSubDataset(grouped_columns, data_type))
    return subdatasets


def train_test_split(
    dataset: Dataset, train_size: float = 0.8
) -> tuple[Dataset, Dataset]:
    data = dataset.to_numpy()
    train_size = int(data.shape[0] * train_size)
    train_dataset = data[:train_size]
    test_dataset = data[train_size:]
    train_dataset = Dataset.from_data_and_metadata(
        train_dataset, dataset.get_metadata()
    )
    test_dataset = Dataset.from_data_and_metadata(test_dataset, dataset.get_metadata())
    return train_dataset, test_dataset
