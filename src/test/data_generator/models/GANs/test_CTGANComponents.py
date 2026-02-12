import pytest

from sdg_core_lib.dataset.datasets import Table
from sdg_core_lib.data_generator.models.GANs.implementation.CTGAN import CTGAN
from sdg_core_lib.data_generator.models.GANs.CTGANComponents import (
    CTGANGenerator,
    CTGANCritic,
)
from sdg_core_lib.data_generator.models.TrainingInfo import TrainingInfo

import os
import shutil


@pytest.fixture()
def model():
    return CTGAN(
        metadata={},
        model_name="CTGAN",
        input_shape="(13,)",
        load_path=None,
    )


@pytest.fixture()
def metadata_builder():
    return [
        {
            "feature_name": "alcohol",
            "feature_position": f"{i}",
            "is_categorical": "false",
            "type": "float64",
            "feature_type": "continuous",
            "feature_size": f"{i}",
        }
        for i in range(2)
    ]


@pytest.fixture()
def model_data_no_load():
    return {
        "metadata": [{}],
        "model_name": "example_model",
        "input_shape": "(13,)",
        "load_path": None,
        "epochs": 1,
    }


@pytest.fixture()
def model_with_metadata(metadata_builder):
    return {
        "metadata": metadata_builder,
        "model_name": "example_model",
        "input_shape": "(13,)",
        "load_path": None,
        "epochs": 1,
    }


@pytest.fixture()
def single_metadata(model_data_no_load):
    model_data_no_load["metadata"] = [
        {
            "feature_name": "A",
            "feature_position": 0,
            "is_categorical": "false",
            "type": "float32",
            "feature_type": "continuous",
            "feature_size": 1,
        },
    ]
    return model_data_no_load


@pytest.fixture()
def data():
    return Table.from_json(
        [
            {
                "column_name": "A",
                "column_type": "continuous",
                "column_datatype": "float32",
                "column_data": [1.0, 2.0, 3.0, 4.0, 5.0],
            },
            {
                "column_name": "B",
                "column_type": "continuous",
                "column_datatype": "float32",
                "column_data": [1.0, 0.0, 1.0, 0.0, 1.0],
            },
        ]
    )


@pytest.fixture()
def data_cat():
    return Table.from_json(
        [
            {
                "column_name": "A",
                "column_type": "continuous",
                "column_datatype": "float32",
                "column_data": [1.0, 2.0, 3.0, 4.0, 5.0],
            },
            {
                "column_name": "B",
                "column_type": "continuous",
                "column_datatype": "float32",
                "column_data": [1.0, 0.0, 1.0, 0.0, 1.0],
            },
            {
                "column_name": "C",
                "column_type": "categorical",
                "column_datatype": "float32",
                "column_data": [0.0, 0.0, 0.0, 0.0, 1.0],
            },
        ]
    )


@pytest.fixture()
def single_metadata_with_cat(model_data_no_load):
    model_data_no_load["metadata"] = [
        {
            "feature_name": "A",
            "feature_position": 0,
            "is_categorical": "false",
            "type": "float32",
            "feature_type": "continuous",
            "feature_size": 2,
        },
        {
            "feature_name": "B",
            "feature_position": 1,
            "is_categorical": "true",
            "type": "float32",
            "feature_type": "categorical",
            "feature_size": 1,
        },
    ]
    return model_data_no_load


def test_instantiate_no_valid_metadata(model_data_no_load):
    with pytest.raises(AttributeError) as exception_info:
        model = CTGAN(**model_data_no_load)
    assert "CTGAN needs a data schema in order to work" in str(exception_info)


def test_self_description(single_metadata_with_cat):
    model = CTGAN(**single_metadata_with_cat)
    self_description = model.self_describe()
    assert self_description is not None
    assert (
        self_description["algorithm"]["name"]
        == "sdg_core_lib.data_generator.models.GANs.implementation.CTGAN.CTGAN"
    ), print(self_description["algorithm"]["name"])
    assert self_description["algorithm"]["default_loss_function"] == "Mean"
    assert (
        self_description["algorithm"]["description"]
        == "A Conditional Tabular Generative Adversarial Network for data generation"
    ), print(self_description["algorithm"]["description"])
    assert self_description["datatypes"] == [
        {"type": "float32", "is_categorical": False},
        {"type": "int32", "is_categorical": False},
        {"type": "int32", "is_categorical": True},
        {"type": "str", "is_categorical": True},
    ]


def test_save(single_metadata_with_cat):
    model = CTGAN(**single_metadata_with_cat)
    model_path = "./test_model"
    os.mkdir(model_path)
    model.save(model_path)
    assert os.path.isfile(os.path.join(model_path, "generator.keras"))
    assert os.path.isfile(os.path.join(model_path, "critic.keras"))
    shutil.rmtree(model_path)


def test_instantiate_with_data(single_metadata_with_cat):
    model = CTGAN(**single_metadata_with_cat)
    assert model._model is not None
    assert model.model_name == single_metadata_with_cat["model_name"]
    assert model._load_path is None
    assert model.input_shape == (13,)
    assert model._epochs == 1
    assert isinstance(model._model.generator, CTGANGenerator)
    assert isinstance(model._model.critic, CTGANCritic)


def test_train_bad_feature_size(single_metadata, data):
    with pytest.raises(AttributeError) as exception_info:
        model = CTGAN(**single_metadata)
    assert "Continous column after normalization must have" in str(exception_info)


def test_train_bad_no_cat_col(single_metadata, data):
    single_metadata["metadata"][0]["feature_size"] = 2
    with pytest.raises(AttributeError) as exception_info:
        model = CTGAN(**single_metadata)
    assert "At least a categorical column must" in str(exception_info)


def test_train_correct(single_metadata_with_cat, data_cat):
    model = CTGAN(**single_metadata_with_cat)
    assert model.training_info is None
    model.train(data_cat.get_computing_data())
    assert isinstance(model.training_info, TrainingInfo)


# TODO: This tests are needed
def test_set_hyperparameters(model):
    hyperparams_wrong = {"wrong": 0.01, "test": 32, "foobar": 10}
    model.set_hyperparameters(**hyperparams_wrong)
    assert model._learning_rate is None
    assert model._batch_size is None
    assert model._epochs is None

    hyperparams = {"learning_rate": 0.01, "batch_size": 32, "epochs": 10}
    model.set_hyperparameters(**hyperparams)
    assert model._learning_rate == 0.01
    assert model._batch_size == 32
    assert model._epochs == 10


def test_infer(model):
    with pytest.raises(AttributeError) as exception_info:
        model.infer(2)
    assert exception_info.type is AttributeError
