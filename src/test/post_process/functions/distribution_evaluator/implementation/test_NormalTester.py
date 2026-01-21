import pytest
import numpy as np

from sdg_core_lib.post_process.functions.distribution_evaluator.implementation.NormalTester import (
    NormalTester,
)


@pytest.fixture
def instance():
    params = [
        {"name": "mean", "value": "0.0", "parameter_type": "float"},
        {"name": "standard_deviation", "value": "1.0", "parameter_type": "float"},
        {"name": "pippo", "value": "pluto", "parameter_type": "str"},
    ]
    return NormalTester.from_json(json_params=params)


def test_check_parameters(instance):
    assert instance.mean == 0.0
    assert isinstance(instance.mean, float)
    assert instance.standard_deviation == 1.0
    assert isinstance(instance.standard_deviation, float)
    assert "pippo" not in instance.__dict__


def test_apply(instance):
    data = np.random.normal(instance.mean, instance.standard_deviation, 100000)
    result = instance.apply(n_rows=None, data=data)
    assert data.shape == (100000,)
    assert result


def test_apply_wrong(instance):
    wrong_data = np.random.normal(5, 1, 100000)
    wrong_data_2 = np.random.normal(0, 10, 100000)
    assert not instance.apply(n_rows=None, data=wrong_data)
    assert not instance.apply(n_rows=None, data=wrong_data_2)
