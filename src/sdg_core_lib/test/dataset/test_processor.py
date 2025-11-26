import os

import numpy as np
import pytest
import shutil
from sdg_core_lib.dataset.datasets import Table, TableProcessor
from sdg_core_lib.dataset.columns import Numeric, Categorical, Column

current_folder = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(current_folder, "outputs")

@pytest.fixture()
def temp_folder():
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

@pytest.fixture()
def teardown():
    yield
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)


def test_process(temp_folder):
    processor = TableProcessor(output_folder)
    col1 = Numeric("col1", "int", 0, np.array([1, 2, 3]).reshape(-1, 1), "continuous")
    col2 = Categorical(
        "col2", "int", 1, np.array([4, 5, 6]).reshape(-1, 1), "categorical"
    )
    columns = [col1, col2]
    table = Table(columns, processor)
    processed_columns = processor.process(table.columns)
    assert len(processed_columns) == 2
    assert isinstance(processed_columns[0], Numeric)
    assert isinstance(processed_columns[1], Categorical)


def test_inverse_process(temp_folder, teardown):
    processor = TableProcessor(output_folder)
    col1 = Numeric("col1", "int", 0, np.array([1, 2, 3]).reshape(-1, 1), "continuous")
    col2 = Categorical(
        "col2", "int", 1, np.array([[1,0,0],[0,1,0],[0,0,1]]).reshape(-1, 3), "categorical"
    )
    columns = [col1, col2]
    processed_columns = processor.inverse_process(columns)
    assert len(processed_columns) == 2
    assert isinstance(processed_columns[0], Column)
    assert isinstance(processed_columns[1], Categorical)


def test_save_and_load(temp_folder, teardown):
    processor = TableProcessor("./outputs")
    col1 = Numeric("col1", "int", 0, np.array([1, 2, 3]).reshape(-1, 1), "continuous")
    col2 = Categorical(
        "col2", "int", 1, np.array([[1,0,0],[0,1,0],[0,0,1]]).reshape(-1, 3), "categorical"
    )
    columns = [col1, col2]
    table = Table(columns, processor)
    pre_table = table.preprocess()
    filenames = os.listdir("./outputs")
    assert len(filenames) == 2
    assert any(["scaler" in filename for filename in filenames])
    assert any(["encoder" in filename for filename in filenames])
    post_table = pre_table.postprocess()
    assert np.all(post_table.columns[0].values == table.columns[0].values)
    assert np.all(post_table.columns[1].values == table.columns[1].values)

