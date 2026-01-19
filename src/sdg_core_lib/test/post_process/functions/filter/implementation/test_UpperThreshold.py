import pytest
import numpy as np

from sdg_core_lib.post_process.functions.filter.implementation.UpperThreshold import (
    UpperThreshold,
)


@pytest.fixture
def instance_strict():
    params = [
        {"name": "value", "value": "5.0", "parameter_type": "float"},
        {"name": "strict", "value": "True", "parameter_type": "bool"},
        {"name": "extra_param", "value": "ignored", "parameter_type": "str"},
    ]
    return UpperThreshold.from_json(json_params=params)


@pytest.fixture
def instance_non_strict():
    params = [
        {"name": "value", "value": "5.0", "parameter_type": "float"},
        {"name": "strict", "value": "False", "parameter_type": "bool"},
    ]
    return UpperThreshold.from_json(json_params=params)


def test_check_parameters(instance_strict):
    assert instance_strict.value == 5.0
    assert isinstance(instance_strict.value, float)
    assert instance_strict.strict is True
    assert isinstance(instance_strict.strict, bool)
    assert "extra_param" not in instance_strict.__dict__


def test_apply_strict(instance_strict):
    # Test with strict=True (<= threshold)
    data = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    filtered_data, indexes = instance_strict.apply(n_rows=len(data), data=data)

    # Should include values <= 5.0
    expected_data = np.array([1.0, 3.0, 5.0])
    expected_indexes = np.array([True, True, True, False, False])

    np.testing.assert_array_equal(filtered_data, expected_data)
    np.testing.assert_array_equal(indexes, expected_indexes)


def test_apply_non_strict(instance_non_strict):
    # Test with strict=False (< threshold)
    data = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    filtered_data, indexes = instance_non_strict.apply(n_rows=len(data), data=data)

    # Should include values < 5.0
    expected_data = np.array([1.0, 3.0])
    expected_indexes = np.array([True, True, False, False, False])

    np.testing.assert_array_equal(filtered_data, expected_data)
    np.testing.assert_array_equal(indexes, expected_indexes)


def test_apply_edge_cases(instance_strict):
    # Test with empty array
    empty_data = np.array([])
    filtered_data, indexes = instance_strict.apply(n_rows=0, data=empty_data)
    assert filtered_data.shape == (0,)
    assert indexes.shape == (0,)

    # Test with all values above threshold
    high_data = np.array([6.0, 7.0, 8.0])
    filtered_data, indexes = instance_strict.apply(
        n_rows=len(high_data), data=high_data
    )
    assert filtered_data.shape == (0,)
    assert np.all(not indexes)

    # Test with all values below threshold
    low_data = np.array([1.0, 2.0, 3.0])
    filtered_data, indexes = instance_strict.apply(n_rows=len(low_data), data=low_data)
    np.testing.assert_array_equal(filtered_data, low_data)
    assert np.all(indexes)
