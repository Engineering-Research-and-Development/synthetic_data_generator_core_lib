import pytest
import numpy as np

from sdg_core_lib.post_process.functions.modification.implementation.BurstNoiseAdder import (
    BurstNoiseAdder,
)


@pytest.fixture
def instance():
    params = [
        {"name": "magnitude", "value": "10.0", "parameter_type": "float"},
        {"name": "n_bursts", "value": "2", "parameter_type": "int"},
        {"name": "burst_duration", "value": "3", "parameter_type": "int"},
        {"name": "extra_param", "value": "ignored", "parameter_type": "str"},
    ]
    return BurstNoiseAdder.from_json(json_params=params)


def test_check_parameters(instance):
    assert instance.magnitude == 10.0
    assert isinstance(instance.magnitude, float)
    assert instance.n_bursts == 2
    assert isinstance(instance.n_bursts, int)
    assert instance.burst_duration == 3
    assert isinstance(instance.burst_duration, int)
    assert "extra_param" not in instance.__dict__


def test_apply(instance):
    # Create test data
    original_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    modified_data, affected_indices, success = instance.apply(
        n_rows=len(original_data), data=original_data
    )

    # Check that data shape is preserved
    assert modified_data.shape == original_data.shape
    assert success is True

    # Check that affected_indices correspond to modified values
    differences = np.abs(modified_data - original_data)
    for idx in affected_indices:
        assert differences[idx] == 10.0  # Should be exactly the magnitude

    # Check that non-affected indices are unchanged
    non_affected_mask = np.ones(len(original_data), dtype=bool)
    non_affected_mask[affected_indices] = False
    assert np.all(differences[non_affected_mask] == 0)


def test_apply_edge_cases(instance):
    # Test with burst_duration > data length
    short_data = np.array([1.0, 2.0])
    result, affected_indices, success = instance.apply(
        n_rows=len(short_data), data=short_data
    )
    # Should return original data unchanged
    np.testing.assert_array_equal(result, short_data)
    assert success is False

    # Test with n_bursts > len(data) // 2
    instance.n_bursts = 10
    medium_data = np.array([1.0, 2.0, 3.0, 4.0])
    result, affected_indices, success = instance.apply(
        n_rows=len(medium_data), data=medium_data
    )
    # Should return original data unchanged
    np.testing.assert_array_equal(result, medium_data)
    assert success is False


def test_check_parameters_invalid():
    # Test with n_bursts < 1
    params = [
        {"name": "magnitude", "value": "10.0", "parameter_type": "float"},
        {"name": "n_bursts", "value": "0", "parameter_type": "int"},
        {"name": "burst_duration", "value": "3", "parameter_type": "int"},
    ]
    with pytest.raises(ValueError, match="Number of bursts must be at least 1"):
        BurstNoiseAdder.from_json(json_params=params)

    # Test with burst_duration < 1
    params = [
        {"name": "magnitude", "value": "10.0", "parameter_type": "float"},
        {"name": "n_bursts", "value": "2", "parameter_type": "int"},
        {"name": "burst_duration", "value": "0", "parameter_type": "int"},
    ]
    with pytest.raises(ValueError, match="Burst duration must be at least 1"):
        BurstNoiseAdder.from_json(json_params=params)
