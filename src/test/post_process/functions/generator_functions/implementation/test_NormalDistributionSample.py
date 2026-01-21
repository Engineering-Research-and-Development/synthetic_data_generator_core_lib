import pytest
import numpy as np

from sdg_core_lib.post_process.functions.generation.implementation.NormalDistributionSample import (
    NormalDistributionSample,
)


@pytest.fixture
def instance():
    params = [
        {"name": "mean", "value": "5.0", "parameter_type": "float"},
        {"name": "standard_deviation", "value": "2.0", "parameter_type": "float"},
        {"name": "extra_param", "value": "ignored", "parameter_type": "str"},
    ]
    return NormalDistributionSample.from_json(json_params=params)


def test_check_parameters(instance):
    assert instance.mean == 5.0
    assert isinstance(instance.mean, float)
    assert instance.standard_deviation == 2.0
    assert isinstance(instance.standard_deviation, float)
    assert "extra_param" not in instance.__dict__


def test_apply(instance):
    n_rows = 1000
    result = instance.apply(n_rows=n_rows, data=np.array([]))

    # Check shape
    assert result.shape == (n_rows, 1)

    # Check statistical properties (within reasonable tolerance)
    sample_mean = np.mean(result)
    sample_std = np.std(result)

    # Mean should be close to target (within 3 standard errors)
    standard_error = instance.standard_deviation / np.sqrt(n_rows)
    assert abs(sample_mean - instance.mean) < 3 * standard_error

    # Standard deviation should be close to target (within 10%)
    assert (
        abs(sample_std - instance.standard_deviation)
        < 0.1 * instance.standard_deviation
    )


def test_apply_edge_cases(instance):
    # Test with zero mean and unit standard deviation
    params = [
        {"name": "mean", "value": "0.0", "parameter_type": "float"},
        {"name": "standard_deviation", "value": "1.0", "parameter_type": "float"},
    ]
    instance = NormalDistributionSample.from_json(json_params=params)
    result = instance.apply(n_rows=100, data=np.array([]))
    assert result.shape == (100, 1)


def test_check_parameters_invalid_std():
    # Test with zero standard deviation
    params = [
        {"name": "mean", "value": "0.0", "parameter_type": "float"},
        {"name": "standard_deviation", "value": "-1.0", "parameter_type": "float"},
    ]
    with pytest.raises(ValueError, match="Standard Deviation cannot be less than 0"):
        NormalDistributionSample.from_json(json_params=params)
