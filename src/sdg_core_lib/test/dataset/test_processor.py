import pytest
from sdg_core_lib.dataset.datasets import Table, TableProcessor
from sdg_core_lib.dataset.columns import Numeric, Categorical, Column


def test_process(self):
    processor = TableProcessor()
    col1 = Column("col1", "int", 0, [1, 2, 3])
    col2 = Column("col2", "int", 1, [4, 5, 6])
    columns = [col1, col2]
    table = Table(columns, processor)
    processed_columns = processor.process(table.columns)
    assert len(processed_columns) == 2
    assert isinstance(processed_columns[0], Numeric)
    assert isinstance(processed_columns[1], Numeric)

def test_inverse_process(self):
    processor = TableProcessor()
    col1 = Numeric("col1", 0, [1, 2, 3])
    col2 = Numeric("col2", 1, [4, 5, 6])
    columns = [col1, col2]
    processed_columns = processor.inverse_process(columns)
    assert len(processed_columns) == 2
    assert isinstance(processed_columns[0], Column)
    assert isinstance(processed_columns[1], Column)