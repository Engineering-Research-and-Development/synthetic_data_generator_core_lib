import pytest
import numpy as np

from sdg_core_lib.post_process.functions.generation.implementation.LinearFunction import (
    LinearFunction,
)


@pytest.fixture
def instance():
    params = [
        {"name": "m", "value": "2.0", "parameter_type": "float"},
        {"name": "q", "value": "1.0", "parameter_type": "float"},
        {"name": "min_value", "value": "0.0", "parameter_type": "float"},
        {"name": "max_value", "value": "10.0", "parameter_type": "float"},
        {"name": "extra_param", "value": "ignored", "parameter_type": "str"},
    ]
    return LinearFunction.from_json(json_params=params)


def test_check_parameters(instance):
    assert instance.m == 2.0
    assert isinstance(instance.m, float)
    assert instance.q == 1.0
    assert isinstance(instance.q, float)
    assert instance.min_value == 0.0
    assert isinstance(instance.min_value, float)
    assert instance.max_value == 10.0
    assert isinstance(instance.max_value, float)
    assert "extra_param" not in instance.__dict__


def test_apply(instance):
    n_rows = 100
    result = instance.apply(n_rows=n_rows, data=np.array([]))

    # Check shape
    assert result.shape == (n_rows, 1)

    # Check that the function follows y = mx + q
    expected_x = np.linspace(instance.min_value, instance.max_value, n_rows)
    expected_y = instance.m * expected_x + instance.q
    np.testing.assert_array_almost_equal(result.flatten(), expected_y)


def test_apply_edge_cases(instance):
    # Test with zero slope
    params = [
        {"name": "m", "value": "0.0", "parameter_type": "float"},
        {"name": "q", "value": "5.0", "parameter_type": "float"},
        {"name": "min_value", "value": "0.0", "parameter_type": "float"},
        {"name": "max_value", "value": "1.0", "parameter_type": "float"},
    ]
    instance = LinearFunction.from_json(json_params=params)
    result = instance.apply(n_rows=10, data=np.array([]))
    np.testing.assert_array_equal(result, 5.0)


def test_check_parameters_invalid_boundary():
    # Test with min_value > max_value
    params = [
        {"name": "m", "value": "1.0", "parameter_type": "float"},
        {"name": "q", "value": "0.0", "parameter_type": "float"},
        {"name": "min_value", "value": "10.0", "parameter_type": "float"},
        {"name": "max_value", "value": "5.0", "parameter_type": "float"},
    ]
    with pytest.raises(ValueError):
        LinearFunction.from_json(json_params=params)
