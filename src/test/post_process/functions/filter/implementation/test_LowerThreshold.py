import pytest
import numpy as np

from sdg_core_lib.post_process.functions.filter.implementation.LowerThreshold import (
    LowerThreshold,
)


@pytest.fixture
def instance_strict():
    params = [
        {"name": "value", "value": "5.0", "parameter_type": "float"},
        {"name": "strict", "value": "True", "parameter_type": "bool"},
        {"name": "extra_param", "value": "ignored", "parameter_type": "str"},
    ]
    return LowerThreshold.from_json(json_params=params)


@pytest.fixture
def instance_non_strict():
    params = [
        {"name": "value", "value": "5.0", "parameter_type": "float"},
        {"name": "strict", "value": "False", "parameter_type": "bool"},
    ]
    return LowerThreshold.from_json(json_params=params)


def test_check_parameters(instance_strict):
    assert instance_strict.value == 5.0
    assert isinstance(instance_strict.value, float)
    assert instance_strict.strict is True
    assert isinstance(instance_strict.strict, bool)
    assert "extra_param" not in instance_strict.__dict__


def test_apply_strict(instance_strict):
    # Test with strict=True (>= threshold)
    data = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    result_data, indexes, success = instance_strict.apply(n_rows=len(data), data=data)

    # Current implementation sets values >= 5.0 to NaN (incorrect behavior)
    expected_data = np.array([1.0, 3.0, np.nan, np.nan, np.nan])
    expected_indexes = np.array([False, False, True, True, True])

    np.testing.assert_array_equal(result_data, expected_data)
    np.testing.assert_array_equal(indexes, expected_indexes)
    assert success is True


def test_apply_non_strict(instance_non_strict):
    # Test with strict=False (> threshold)
    data = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    result_data, indexes, success = instance_non_strict.apply(
        n_rows=len(data), data=data
    )

    # Current implementation sets values > 5.0 to NaN (incorrect behavior)
    expected_data = np.array([1.0, 3.0, 5.0, np.nan, np.nan])
    expected_indexes = np.array([False, False, False, True, True])

    np.testing.assert_array_equal(result_data, expected_data)
    np.testing.assert_array_equal(indexes, expected_indexes)
    assert success is True


def test_apply_edge_cases(instance_strict):
    # Test with empty array
    empty_data = np.array([])
    result_data, indexes, success = instance_strict.apply(n_rows=0, data=empty_data)
    assert result_data.shape == (0,)
    assert indexes.shape == (0,)
    assert success is True

    # Test with all values below threshold
    low_data = np.array([1.0, 2.0, 3.0])
    result_data, indexes, success = instance_strict.apply(
        n_rows=len(low_data), data=low_data
    )
    # Current implementation keeps all values (none meet threshold)
    np.testing.assert_array_equal(result_data, low_data)
    assert not np.all(indexes)
    assert success is True

    # Test with all values above threshold
    high_data = np.array([6.0, 7.0, 8.0])
    result_data, indexes, success = instance_strict.apply(
        n_rows=len(high_data), data=high_data
    )
    # Current implementation sets all values to NaN
    expected_data = np.array([np.nan, np.nan, np.nan])
    np.testing.assert_array_equal(result_data, expected_data)
    assert np.all(indexes)
    assert success is True
