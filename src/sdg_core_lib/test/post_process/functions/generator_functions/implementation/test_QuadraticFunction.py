import pytest
import numpy as np

from sdg_core_lib.post_process.functions.generation.implementation.QuadraticFunction import (
    QuadraticFunction,
)


@pytest.fixture
def instance():
    params = [
        {"name": "a", "value": "2.0", "parameter_type": "float"},
        {"name": "b", "value": "-1.0", "parameter_type": "float"},
        {"name": "c", "value": "3.0", "parameter_type": "float"},
        {"name": "min_value", "value": "-5.0", "parameter_type": "float"},
        {"name": "max_value", "value": "5.0", "parameter_type": "float"},
        {"name": "extra_param", "value": "ignored", "parameter_type": "str"},
    ]
    return QuadraticFunction.from_json(json_params=params)


def test_check_parameters(instance):
    assert instance.a == 2.0
    assert isinstance(instance.a, float)
    assert instance.b == -1.0
    assert isinstance(instance.b, float)
    assert instance.c == 3.0
    assert isinstance(instance.c, float)
    assert instance.min_value == -5.0
    assert isinstance(instance.min_value, float)
    assert instance.max_value == 5.0
    assert isinstance(instance.max_value, float)
    assert "extra_param" not in instance.__dict__


def test_apply(instance):
    n_rows = 100
    result = instance.apply(n_rows=n_rows, data=np.array([]))

    # Check shape
    assert result.shape == (n_rows, 1)

    # Check that the function follows y = ax^2 + bx + c
    expected_x = np.linspace(instance.min_value, instance.max_value, n_rows)
    expected_y = instance.a * expected_x**2 + instance.b * expected_x + instance.c
    np.testing.assert_array_almost_equal(result.flatten(), expected_y)


def test_apply_edge_cases(instance):
    # Test with a = 0 (should reduce to linear function)
    params = [
        {"name": "a", "value": "0.0", "parameter_type": "float"},
        {"name": "b", "value": "2.0", "parameter_type": "float"},
        {"name": "c", "value": "1.0", "parameter_type": "float"},
        {"name": "min_value", "value": "0.0", "parameter_type": "float"},
        {"name": "max_value", "value": "1.0", "parameter_type": "float"},
    ]
    instance = QuadraticFunction.from_json(json_params=params)
    result = instance.apply(n_rows=10, data=np.array([]))
    expected_x = np.linspace(0.0, 1.0, 10)
    expected_y = 2.0 * expected_x + 1.0
    np.testing.assert_array_almost_equal(result.flatten(), expected_y)


def test_check_parameters_invalid_boundary():
    # Test with min_value > max_value
    params = [
        {"name": "a", "value": "1.0", "parameter_type": "float"},
        {"name": "b", "value": "0.0", "parameter_type": "float"},
        {"name": "c", "value": "0.0", "parameter_type": "float"},
        {"name": "min_value", "value": "10.0", "parameter_type": "float"},
        {"name": "max_value", "value": "5.0", "parameter_type": "float"},
    ]
    with pytest.raises(ValueError):
        QuadraticFunction.from_json(json_params=params)
