import numpy as np
from sdg_core_lib.dataset.datasets import Table, TableProcessor
from sdg_core_lib.dataset.columns import Numeric, Categorical, Column


def test_process():
    processor = TableProcessor("./outputs")
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


def test_inverse_process():
    processor = TableProcessor("./outputs")
    col1 = Numeric("col1", "int", 0, np.array([1, 2, 3]).reshape(-1, 1), "continuous")
    col2 = Categorical(
        "col2", "int", 1, np.array([[1,0,0],[0,1,0],[0,0,1]]).reshape(-1, 3), "categorical"
    )
    columns = [col1, col2]
    processed_columns = processor.inverse_process(columns)
    assert len(processed_columns) == 2
    assert isinstance(processed_columns[0], Column)
    assert isinstance(processed_columns[1], Categorical)
