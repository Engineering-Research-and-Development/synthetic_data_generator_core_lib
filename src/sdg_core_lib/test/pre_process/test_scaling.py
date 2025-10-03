import pytest
import numpy as np

from sdg_core_lib.preprocess.scale import (
    standardize_simple_tabular_time_series,
    standardize_simple_tabular_input,
)


@pytest.fixture()
def correct_tabular_input():
    return np.linspace(-10, 10, 100).reshape(10, 10)


@pytest.fixture()
def correct_time_series_input():
    return np.linspace(-10, 10, 1000).reshape(10, 2, 50)


def test_correct_tabular_scaling(correct_tabular_input):
    scaler, standardized_train_data, standardized_test_data = (
        standardize_simple_tabular_input(
            train_data=correct_tabular_input, test_data=correct_tabular_input
        )
    )

    assert type(standardized_train_data) is np.ndarray
    assert standardized_train_data.shape == correct_tabular_input.shape
    assert standardized_test_data.shape == correct_tabular_input.shape


def test_incorrect_tabular_scaling(correct_time_series_input):
    with pytest.raises(ValueError) as exception_info:
        _, _, _ = standardize_simple_tabular_input(train_data=correct_time_series_input)
    assert exception_info.type is ValueError


def test_correct_time_series_scaling(correct_time_series_input):
    scaler, standardized_train_data, standardized_test_data = (
        standardize_simple_tabular_time_series(
            train_data=correct_time_series_input, test_data=correct_time_series_input
        )
    )

    assert type(standardized_train_data) is np.ndarray
    assert standardized_train_data.shape == correct_time_series_input.shape
    assert standardized_test_data.shape == correct_time_series_input.shape


def test_incorrect_time_series_scaling(correct_tabular_input):
    with pytest.raises(ValueError) as exception_info:
        _, _, _ = standardize_simple_tabular_time_series(
            train_data=correct_tabular_input
        )
    assert exception_info.type is ValueError
