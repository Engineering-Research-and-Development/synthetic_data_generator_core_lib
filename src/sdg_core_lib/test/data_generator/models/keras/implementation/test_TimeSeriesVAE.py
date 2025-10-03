import numpy as np
import pytest
import os
import shutil
from sklearn.preprocessing import MinMaxScaler

from sdg_core_lib.NumericDataset import NumericDataset
from sdg_core_lib.data_generator.models.TrainingInfo import TrainingInfo
from sdg_core_lib.data_generator.models.keras.VAE import VAE
from sdg_core_lib.data_generator.models.keras.implementation.TimeSeriesVAE import (
    TimeSeriesVAE,
)


@pytest.fixture()
def model_data_correct_train():
    return {
        "metadata": {"example_key": "example_value"},
        "model_name": "example_model",
        "input_shape": "(2, 51)",
        "load_path": None,
        "epochs": 1,
    }


@pytest.fixture()
def data():
    return NumericDataset(
        [
            {
                "column_name": "A",
                "column_type": "time_series",
                "column_datatype": "float64",
                "column_data": np.linspace(-10, 10, 1020).reshape(-1, 51).tolist(),
            },
            {
                "column_name": "B",
                "column_type": "time_series",
                "column_datatype": "float64",
                "column_data": np.linspace(-10, 10, 1020).reshape(-1, 51).tolist(),
            },
        ]
    )


def test_instantiate(model_data_correct_train):
    model = TimeSeriesVAE(**model_data_correct_train)
    assert model.model_name == model_data_correct_train["model_name"]
    assert model._load_path is None
    assert model.input_shape == (2, 51)
    assert model._epochs == 1
    assert type(model._model) is VAE
    assert model._scaler is None


def test_preprocess(model_data_correct_train, data):
    model = TimeSeriesVAE(**model_data_correct_train)
    assert model._scaler is None
    scaled_data = model._pre_process(data)
    assert model._scaler is not None and type(model._scaler) is MinMaxScaler
    assert type(scaled_data) is np.ndarray
    assert scaled_data.shape == data.get_numpy_data(data.dataframe).shape
    assert scaled_data.shape[1:] == model.input_shape


def test_train_correct(model_data_correct_train, data):
    model = TimeSeriesVAE(**model_data_correct_train)
    assert model.training_info is None
    assert model._scaler is None
    model.train(data)
    assert type(model._scaler) is MinMaxScaler
    assert type(model.training_info) is TrainingInfo


def test_save(model_data_correct_train):
    model = TimeSeriesVAE(**model_data_correct_train)
    model_path = "./test_model"
    os.mkdir(model_path)
    model.save(model_path)
    assert os.path.isfile(os.path.join(model_path, "encoder.keras"))
    assert os.path.isfile(os.path.join(model_path, "decoder.keras"))
    assert os.path.isfile(os.path.join(model_path, "scaler.skops"))
    shutil.rmtree(model_path)


def test_self_description(model_data_correct_train):
    model = TimeSeriesVAE(**model_data_correct_train)
    self_description = model.self_describe()
    assert self_description is not None
    assert (
        self_description["algorithm"]["name"]
        == "sdg_core_lib.data_generator.models.keras.implementation.TimeSeriesVAE.TimeSeriesVAE"
    )
    assert self_description["algorithm"]["default_loss_function"] == "ELBO LOSS"
    assert (
        self_description["algorithm"]["description"]
        == "A Beta-Variational Autoencoder for time series generation"
    )
    assert self_description["datatypes"] == [
        {"type": "float32", "is_categorical": False},
        {"type": "int32", "is_categorical": False},
        {"type": "int64", "is_categorical": False},
    ]


def test_infer(model_data_correct_train, data):
    n_rows = 2
    model = TimeSeriesVAE(**model_data_correct_train)
    results = model.infer(n_rows)
    assert results.shape == (n_rows, *model.input_shape)
