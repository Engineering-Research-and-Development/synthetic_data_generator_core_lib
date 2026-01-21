import pytest
import numpy as np

from sdg_core_lib.post_process.functions.generation.implementation.SinusoidalFunction import (
    SinusoidalFunction,
)


@pytest.fixture
def instance():
    params = [
        {"name": "a", "value": "2.0", "parameter_type": "float"},
        {"name": "f", "value": "1.0", "parameter_type": "float"},
        {"name": "phi", "value": "0.25", "parameter_type": "float"},
        {"name": "v", "value": "1.0", "parameter_type": "float"},
        {"name": "min_value", "value": "0.0", "parameter_type": "float"},
        {"name": "max_value", "value": "2.0", "parameter_type": "float"},
        {"name": "extra_param", "value": "ignored", "parameter_type": "str"},
    ]
    return SinusoidalFunction.from_json(json_params=params)


def test_check_parameters(instance):
    assert instance.a == 2.0
    assert isinstance(instance.a, float)
    assert instance.f == 1.0
    assert isinstance(instance.f, float)
    assert instance.phi == 0.25
    assert isinstance(instance.phi, float)
    assert instance.v == 1.0
    assert isinstance(instance.v, float)
    assert instance.min_value == 0.0
    assert isinstance(instance.min_value, float)
    assert instance.max_value == 2.0
    assert isinstance(instance.max_value, float)
    assert "extra_param" not in instance.__dict__


def test_apply(instance):
    n_rows = 100
    result_data, indexes, success = instance.apply(n_rows=n_rows, data=np.array([]))

    # Check shape
    assert result_data.shape == (n_rows, 1)

    # Check that the function follows y = a*sin(2*pi*f*x + 2*pi*phi) + v
    import math

    expected_x = np.linspace(instance.min_value, instance.max_value, n_rows)
    expected_y = (
        instance.a
        * np.sin(2 * math.pi * instance.f * expected_x + 2 * math.pi * instance.phi)
        + instance.v
    )
    np.testing.assert_array_almost_equal(result_data.flatten(), expected_y, decimal=10)
    assert success is True


def test_apply_edge_cases(instance):
    # Test with a = 0 (should be constant function)
    params = [
        {"name": "a", "value": "0.0", "parameter_type": "float"},
        {"name": "f", "value": "1.0", "parameter_type": "float"},
        {"name": "phi", "value": "0.0", "parameter_type": "float"},
        {"name": "v", "value": "5.0", "parameter_type": "float"},
        {"name": "min_value", "value": "0.0", "parameter_type": "float"},
        {"name": "max_value", "value": "1.0", "parameter_type": "float"},
    ]
    instance = SinusoidalFunction.from_json(json_params=params)
    result_data, indexes, success = instance.apply(n_rows=10, data=np.array([]))
    expected_data = np.full((10, 1), 5.0)
    np.testing.assert_array_equal(result_data, expected_data)
    assert success is True


def test_check_parameters_invalid_boundary():
    # Test with min_value > max_value
    params = [
        {"name": "a", "value": "1.0", "parameter_type": "float"},
        {"name": "f", "value": "1.0", "parameter_type": "float"},
        {"name": "phi", "value": "0.0", "parameter_type": "float"},
        {"name": "v", "value": "0.0", "parameter_type": "float"},
        {"name": "min_value", "value": "10.0", "parameter_type": "float"},
        {"name": "max_value", "value": "5.0", "parameter_type": "float"},
    ]
    with pytest.raises(ValueError):
        SinusoidalFunction.from_json(json_params=params)
