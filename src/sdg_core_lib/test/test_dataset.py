import pandas as pd
import pytest
import numpy as np

from sdg_core_lib.NumericDataset import NumericDataset


@pytest.fixture
def correct_dataset():
    return [
        {
            "column_name": "A",
            "column_type": "continuous",
            "column_datatype": "float64",
            "column_data": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        {
            "column_name": "B",
            "column_type": "categorical",
            "column_datatype": "object",
            "column_data": ["a", "b", "c", "d", "e"],
        },
        {
            "column_name": "C",
            "column_type": "continuous",
            "column_datatype": "int64",
            "column_data": [1, 2, 3, 4, 5],
        },
        {
            "column_name": "D",
            "column_type": "test",
            "column_datatype": "int64",
            "column_data": [1, 2, 3, 4, 5],
        },
    ]


@pytest.fixture
def complex_dataset():
    return [
        {
            "column_name": "A",
            "column_type": "time_series",
            "column_datatype": "float64",
            "column_data": [[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]],
        }
    ]


@pytest.fixture
def error_dataset():
    return [
        {
            "column_name": "A",
            "column_type": "categorical",
            "column_datatype": "string",
            "column_data": ["a", "b", "c", "d", "e"],
        }
    ]


@pytest.fixture
def empty_dataset():
    return []


def test_initialization(correct_dataset):
    dataset = NumericDataset(correct_dataset)
    assert len(dataset.columns) == 4
    assert len(dataset.categorical_columns) == 1
    assert len(dataset.continuous_columns) == 2
    assert len(dataset.unrecognized_columns) == 1
    assert dataset.get_numpy_data(dataset.dataframe).shape == (5, 4)


def test_dataset_complexity(complex_dataset):
    dataset = NumericDataset(complex_dataset)
    print(np.array(dataset.dataframe.to_numpy().tolist()).shape)
    assert len(dataset.columns) == 1
    assert len(dataset.categorical_columns) == 0
    assert len(dataset.continuous_columns) == 0
    assert len(dataset.unrecognized_columns) == 1
    assert dataset.get_numpy_data(dataset.dataframe).shape == (2, 1, 5)


def test_error_initialization(error_dataset):
    with pytest.raises(TypeError) as exception_info:
        _ = NumericDataset(error_dataset)
    assert exception_info.type is TypeError


def test_parse_tabular_data_json(correct_dataset):
    dataset = NumericDataset(correct_dataset)
    print(dataset.dataframe["A"].dtype)
    list_dict = dataset.parse_tabular_data_json()
    assert len(list_dict) == len(dataset.columns)
    assert list_dict[0]["column_name"] == "A"
    assert list_dict[0]["column_type"] == "continuous"
    assert list_dict[0]["column_datatype"] == "float64"
    assert list_dict[0]["column_data"] == [1, 2, 3, 4, 5]
    assert list_dict[1]["column_name"] == "B"
    assert list_dict[1]["column_type"] == "categorical"
    assert list_dict[1]["column_datatype"] == "object"
    assert list_dict[1]["column_data"] == ["a", "b", "c", "d", "e"]
    assert list_dict[2]["column_name"] == "C"
    assert list_dict[2]["column_type"] == "continuous"
    assert list_dict[2]["column_datatype"] == "int64"
    assert list_dict[2]["column_data"] == [1, 2, 3, 4, 5]
    assert list_dict[3]["column_name"] == "D"
    assert list_dict[3]["column_type"] == "none"
    assert list_dict[3]["column_datatype"] == "int64"
    assert list_dict[3]["column_data"] == [1, 2, 3, 4, 5]


def test_parse_data_to_registry(correct_dataset):
    dataset = NumericDataset(correct_dataset)
    feature_list = dataset.parse_data_to_registry()
    assert len(feature_list) == len(dataset.columns)
    assert feature_list[0]["feature_name"] == "A"
    assert feature_list[0]["feature_position"] == 0
    assert feature_list[0]["is_categorical"] is False
    assert feature_list[0]["type"] == "float64"
    assert feature_list[1]["feature_name"] == "B"
    assert feature_list[1]["feature_position"] == 1
    assert feature_list[1]["is_categorical"] is True
    assert feature_list[1]["type"] == "object"
    assert feature_list[2]["feature_name"] == "C"
    assert feature_list[2]["feature_position"] == 2
    assert feature_list[2]["is_categorical"] is False
    assert feature_list[2]["type"] == "int64"
    assert feature_list[3]["feature_name"] == "D"
    assert feature_list[3]["feature_position"] == 3
    assert feature_list[3]["is_categorical"] is False
    assert feature_list[3]["type"] == "int64"


def test_get_data(correct_dataset):
    dataset = NumericDataset(correct_dataset)
    dataframe, columns, continuous_columns, categorical_columns = dataset.get_data()
    assert type(dataframe) is pd.DataFrame
    assert columns == ["A", "B", "C", "D"]
    assert continuous_columns == ["A", "C"]
    assert categorical_columns == ["B"]


def test_empty_dataset(empty_dataset):
    with pytest.raises(ValueError) as exception_info:
        _ = NumericDataset(empty_dataset)
    assert exception_info.type is ValueError
