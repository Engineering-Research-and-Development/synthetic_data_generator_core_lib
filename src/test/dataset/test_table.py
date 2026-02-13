import pytest
import os
import json
import shutil
import numpy as np

from sdg_core_lib.dataset.datasets import Table
from sdg_core_lib.preprocess.table_processor import TableProcessor
from sdg_core_lib.preprocess.strategies.vae_strategy import (
    TabularVAEPreprocessingStrategy,
)

current_folder = os.path.dirname(os.path.abspath(__file__))
correct_dataset = json.load(open(os.path.join(current_folder, "correct_dataset.json")))
correct_skeleton = json.load(
    open(os.path.join(current_folder, "correct_skeleton.json"))
)


@pytest.fixture()
def temp_folder():
    folder = os.path.join(current_folder, "temp_folder")
    os.mkdir(folder)
    yield folder
    shutil.rmtree(folder)


def test_table_from_json(temp_folder):
    table = Table.from_json(correct_dataset)
    assert len(table.pk_col_indexes) == 2
    assert len(table.columns) == 4


def test_table_from_skeleton(temp_folder):
    table = Table.from_skeleton(correct_skeleton)
    assert len(table.pk_col_indexes) == 2
    assert len(table.columns) == 4


def test_table_get_primary_keys(temp_folder):
    table = Table.from_json(correct_dataset)
    assert table.get_primary_keys() == [
        table.columns[0],
        table.columns[1],
    ]


def test_table_get_numeric_columns(temp_folder):
    table = Table.from_json(correct_dataset)
    assert table.get_numeric_columns() == [table.columns[2]]


def test_table_get_categorical_columns(temp_folder):
    table = Table.from_json(correct_dataset)
    assert table.get_categorical_columns() == [table.columns[3]]


def test_table_get_computing_data(temp_folder):
    table = Table.from_json(correct_dataset)
    data = table.get_computing_data()
    assert isinstance(data, np.ndarray)
    assert np.all(
        data == np.array([[1, 2, 3, 4, 5, 6], ["a", "b", "c", "d", "e", "f"]]).T
    )


def test_table_get_computing_shape(temp_folder):
    table = Table.from_json(correct_dataset)
    assert table.get_computing_shape() == (6, 2)


def test_table_clone(temp_folder):
    table = Table.from_json(correct_dataset)
    data = np.random.randint(0, 100, size=(4, 2))
    new_table = table.clone(data)
    assert len(new_table.columns) == 4
    assert new_table.get_computing_shape() == (4, 2)
    assert np.all(new_table.get_computing_data() == data)


def test_table_to_json(temp_folder):
    table = Table.from_json(correct_dataset)
    table_json = table.to_json()
    assert isinstance(table_json, list)
    assert len(table_json) == 4
    assert table_json == correct_dataset


def test_table_to_skeleton(temp_folder):
    table = Table.from_json(correct_dataset)
    skeleton = table.to_skeleton()
    assert isinstance(skeleton, list)
    assert len(skeleton) == 4
    assert all(isinstance(col, dict) for col in skeleton)


def test_from_json_invalid_data(temp_folder):
    # Test with missing required fields
    invalid_data = [{"column_name": "test"}]  # Missing column_type and column_data
    with pytest.raises(TypeError):
        Table.from_json(invalid_data)

    # Test with invalid column type
    invalid_type_data = [
        {"column_name": "test", "column_type": "invalid_type", "column_data": [1, 2, 3]}
    ]
    with pytest.raises(TypeError):
        Table.from_json(invalid_type_data)


def test_clone_invalid_shape(temp_folder):
    table = Table.from_json(correct_dataset)
    # Test with incorrect number of columns
    invalid_data = np.random.randint(0, 100, size=(6, 3))  # Expected 2 columns
    with pytest.raises(ValueError):
        table.clone(invalid_data)


def test_self_pk_integrity(temp_folder):
    # Create a table with duplicate PKs
    invalid_data = [
        {
            "column_name": "id1",
            "column_type": "primary_key",
            "column_data": [1, 1, 2, 2],
            "column_datatype": "int",
        },
        {
            "column_name": "id2",
            "column_type": "primary_key",
            "column_data": [1, 1, 1, 2],
            "column_datatype": "int",
        },
        {
            "column_name": "value",
            "column_type": "continuous",
            "column_data": [1.1, 2.2, 3.3, 4.4],
            "column_datatype": "float",
        },
    ]
    table = Table.from_json(invalid_data)
    assert table._self_pk_integrity() is False


def test_empty_table(temp_folder):
    empty_data = []
    with pytest.raises(ValueError):
        Table.from_json(empty_data)


def test_mixed_data_types(temp_folder):
    mixed_data = [
        {
            "column_name": "id",
            "column_type": "primary_key",
            "column_data": [1, 2, 3],
            "column_datatype": "int",
        },
        {
            "column_name": "score",
            "column_type": "continuous",
            "column_data": [1.1, 2.2, 3.3],
            "column_datatype": "float",
        },
        {
            "column_name": "category",
            "column_type": "categorical",
            "column_data": ["a", "b", "c"],
            "column_datatype": "str",
        },
    ]
    table = Table.from_json(mixed_data)
    assert len(table.get_numeric_columns()) == 1
    assert len(table.get_categorical_columns()) == 1
    assert len(table.get_primary_keys()) == 1


def test_duplicate_group_index(temp_folder):
    duplicate_group_data = [
        {
            "column_name": "group1",
            "column_type": "group_index",
            "column_data": [1, 2, 3],
            "column_datatype": "int",
        },
        {
            "column_name": "group2",
            "column_type": "group_index",
            "column_data": [1, 2, 3],
            "column_datatype": "int",
        },
    ]
    with pytest.raises(ValueError, match="Group index already set"):
        Table.from_json(duplicate_group_data)


def test_table_with_missing_values(temp_folder):
    data_with_missing = [
        {
            "column_name": "id",
            "column_type": "primary_key",
            "column_data": [1, 2, 3],
            "column_datatype": "int",
        },
        {
            "column_name": "value",
            "column_type": "continuous",
            "column_data": [1.1, None, 3.3],
            "column_datatype": "float",
        },
    ]
    # TODO: Improve NoneValue Management in future
    table = Table.from_json(data_with_missing)
    assert len(table.columns) == 2


def test_data_aggregation(temp_folder):
    data = [
        {
            "column_name": "category",
            "column_type": "categorical",
            "column_data": ["Test", "Production", "Test", "Test", "Production"],
            "column_datatype": "str",
        },
        {
            "column_name": "value",
            "column_type": "continuous",
            "column_data": [10, 20, 30, 40, 50],
            "column_datatype": "int",
        },
    ]
    table = Table.from_json(data)
    # Test computing data
    computing_data = table.get_computing_data()
    assert len(computing_data) == 5  # 5 rows
    assert len(computing_data[0]) == 2  # 1 computing column (value)


def test_preprocess_continuous_column(temp_folder):
    data = [
        {
            "column_name": "value",
            "column_type": "continuous",
            "column_data": [1, 2, 3],
            "column_datatype": "int",
        }
    ]
    processor = TableProcessor(temp_folder).set_strategy(
        TabularVAEPreprocessingStrategy()
    )
    table = Table.from_json(data)
    preprocessed_table = table.preprocess(processor)
    compared_array = np.array(data[0]["column_data"]).reshape(-1, 1)
    scaled_array = (compared_array - np.mean(compared_array)) / np.std(compared_array)
    assert np.all(preprocessed_table.get_computing_data() == scaled_array)


def test_postprocess_continuous_column(temp_folder):
    data = [
        {
            "column_name": "value",
            "column_type": "continuous",
            "column_data": [1, 2, 3],
            "column_datatype": "int",
        }
    ]
    table = Table.from_json(data)
    processor = TableProcessor(temp_folder).set_strategy(
        TabularVAEPreprocessingStrategy()
    )
    preprocessed_table = table.preprocess(processor)
    postprocessed_table = preprocessed_table.postprocess(processor)
    assert np.all(
        postprocessed_table.get_computing_data() == np.array([1, 2, 3]).reshape(-1, 1)
    )


def test_preprocess_categorical_column(temp_folder):
    data = [
        {
            "column_name": "value",
            "column_type": "categorical",
            "column_data": ["a", "b", "c"],
            "column_datatype": "str",
        }
    ]
    table = Table.from_json(data)
    processor = TableProcessor(temp_folder).set_strategy(
        TabularVAEPreprocessingStrategy()
    )
    processed_table = table.preprocess(processor)
    assert np.all(
        processed_table.get_computing_data()
        == np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )


def test_postprocess_categorical_column(temp_folder):
    data = [
        {
            "column_name": "value",
            "column_type": "categorical",
            "column_data": ["a", "b", "c"],
            "column_datatype": "str",
        }
    ]
    table = Table.from_json(data)
    processor = TableProcessor(temp_folder).set_strategy(
        TabularVAEPreprocessingStrategy()
    )
    processed_table = table.preprocess(processor)
    postprocessed_table = processed_table.postprocess(processor)
    assert np.all(
        postprocessed_table.get_computing_data()
        == np.array(["a", "b", "c"]).reshape(-1, 1)
    )
