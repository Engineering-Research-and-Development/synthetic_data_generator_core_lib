import pytest
import numpy as np

from sdg_core_lib.post_process.functions.modification.implementation.WhiteNoiseAdder import (
    WhiteNoiseAdder,
)


@pytest.fixture
def instance():
    params = [
        {"name": "mean", "value": "0.5", "parameter_type": "float"},
        {"name": "standard_deviation", "value": "1.0", "parameter_type": "float"},
        {"name": "extra_param", "value": "ignored", "parameter_type": "str"},
    ]
    return WhiteNoiseAdder.from_json(json_params=params)


def test_check_parameters(instance):
    assert instance.mean == 0.5
    assert isinstance(instance.mean, float)
    assert instance.standard_deviation == 1.0
    assert isinstance(instance.standard_deviation, float)
    assert "extra_param" not in instance.__dict__


def test_apply(instance):
    # Create test data
    original_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    modified_data, indices, success = instance.apply(
        n_rows=len(original_data), data=original_data
    )

    # Check that data shape is preserved
    assert modified_data.shape == original_data.shape
    assert indices.shape == original_data.shape
    assert success is True

    # Check that modified_data is different from original_data (due to noise)
    assert not np.array_equal(modified_data, original_data)


def test_apply_edge_cases(instance):
    # Test with zero mean and zero standard deviation
    params = [
        {"name": "mean", "value": "0.0", "parameter_type": "float"},
        {"name": "standard_deviation", "value": "0.0", "parameter_type": "float"},
    ]
    instance = WhiteNoiseAdder.from_json(json_params=params)
    original_data = np.array([1.0, 2.0, 3.0])
    modified_data, indices, success = instance.apply(
        n_rows=len(original_data), data=original_data
    )

    # With zero standard deviation, noise should be all zeros
    np.testing.assert_array_equal(modified_data, original_data)
    np.testing.assert_array_equal(indices, np.array([0, 1, 2]))
    assert success is True

    # Test with empty array
    empty_data = np.array([])
    modified_data, indices, success = instance.apply(n_rows=0, data=empty_data)
    assert modified_data.shape == (0,)
    assert indices.shape == (0,)
    assert success is True


def test_apply_large_sample():
    # Test with larger sample to better verify statistical properties
    params = [
        {"name": "mean", "value": "2.0", "parameter_type": "float"},
        {"name": "standard_deviation", "value": "0.5", "parameter_type": "float"},
    ]
    instance = WhiteNoiseAdder.from_json(json_params=params)
    original_data = np.ones(1000) * 10.0  # Constant data
    modified_data, indices, success = instance.apply(
        n_rows=len(original_data), data=original_data
    )

    # Check that indices are correct
    expected_indices = np.array(range(len(original_data)))
    np.testing.assert_array_equal(indices, expected_indices)
    assert success is True

    # Calculate the noise that was added
    noise = modified_data - original_data

    # Check statistical properties more precisely with larger sample
    noise_mean = np.mean(noise)
    noise_std = np.std(noise)

    # Mean should be very close to target (within 3 standard errors for more tolerance)
    standard_error = instance.standard_deviation / np.sqrt(len(noise))
    assert abs(noise_mean - instance.mean) < 3 * standard_error

    # Standard deviation should be close to target (within 10%)
    assert (
        abs(noise_std - instance.standard_deviation) < 0.1 * instance.standard_deviation
    )
