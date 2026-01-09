import keras
import numpy as np
import keras_tuner as kt
from keras import layers
from keras_tuner import HyperParameters

from data_generator.models.TrainingInfo import TrainingInfo
from sdg_core_lib.data_generator.models.ModelInfo import ModelInfo, AllowedData
from sdg_core_lib.data_generator.models.keras.KerasBaseVAE import KerasBaseVAE
from sdg_core_lib.data_generator.models.keras.VAE import Sampling, VAE



class AutoTabularVAE(KerasBaseVAE):
    """
    TabularVAE is a class that implements a Variational Autoencoder (VAE) for tabular data generation.
    It inherits from the KerasBaseVAE class and provides functionality specific to handling tabular data.

    Attributes:
        _latent_dim (int): The dimensionality of the latent space.
        _beta (float): The beta parameter for the VAE loss function.
        _learning_rate (float): Learning rate for the optimizer.
        _batch_size (int): Number of samples per batch during training.
        _epochs (int): Number of training epochs.
        _scaler: Scaler used for standardizing input data.

    Methods:
        __init__: Initializes the TabularVAE with model parameters.
        _load_model: Loads the VAE model with specified encoder and decoder.
        _build: Builds the VAE model architecture.
        _pre_process: Pre-processes input data using standardization.
        self_describe: Provides metadata information about the model.
    """

    def __init__(
        self,
        metadata: dict,
        model_name: str,
        input_shape: str,
        load_path: str | None,
        latent_dim: int = 2,
        learning_rate: float = 1e-3,
        batch_size: int = 8,
        epochs: int = 200,
    ):
        super().__init__(metadata, model_name, input_shape, load_path, latent_dim)
        self._beta = 0.15
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._instantiate()

    def _load_model(self, encoder, decoder):
        self._model = VAE(encoder, decoder, self._beta)

    def _build(self, input_shape: tuple[int, ...]):
        encoder_inputs = keras.Input(shape=input_shape)
        x = layers.Dense(64, activation="relu")(encoder_inputs)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(self._latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self._latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        latent_inputs = keras.Input(shape=(self._latent_dim,))
        y = layers.Dense(16, activation="relu")(latent_inputs)
        y = layers.Dense(128, activation="relu")(y)
        y = layers.Dense(64, activation="relu")(y)
        decoder_outputs = layers.Dense(input_shape[0], activation="linear")(y)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

        vae = VAE(encoder, decoder, self._beta, name="TabularVAE")
        vae.summary()
        return vae

    def _build_automodel(self, input_shape: tuple[int, ...], hp: HyperParameters):
        encoder_layers = hp.Int('encoder_layers', min_value=1, max_value=4)
        decoder_layers = hp.Int('decoder_layers', min_value=1, max_value=4)
        units_multiplier = hp.Choice('units_multiplier', values=[16, 32, 64, 128])
        activation = hp.Choice(f'activation', values=['relu', 'elu', 'selu', 'gelu'])
        beta = hp.Float('beta', min_value=0.0001, max_value=0.01, sampling="log")
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')

        encoder_inputs = keras.Input(shape=input_shape)
        x = layers.Flatten()(encoder_inputs)
        for i in range(encoder_layers):
            units = units_multiplier * (2 ** (encoder_layers - i - 1))
            x = layers.Dense(
                units,
                activation=activation
            )(x)
        z_mean = layers.Dense(self._latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self._latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        latent_inputs = keras.Input(shape=(self._latent_dim,))
        x = latent_inputs

        for i in range(decoder_layers):
            units = units_multiplier * (2 ** i)
            x = layers.Dense(
                units,
                activation=activation
            )(x)
        x = layers.Dense(np.prod(input_shape), activation='linear')(x)
        decoder_outputs = layers.Reshape(input_shape)(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')
        vae = VAE(encoder, decoder, beta=beta)
        vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
        vae.build((None,) + input_shape)
        return vae


    def train(
        self,
        data: np.ndarray,
        learning_rate: float = None,
        batch_size: int = None,
        epochs: int = None,
    ):
        """
        Trains the VAE model on the provided data.

        :param data: the dataset to train the model on, must be a NumericDataset
        :param learning_rate: optional, the learning rate to use for training
        :param batch_size: optional, the batch size to use for training
        :param epochs: optional, the number of epochs to train the model for
        :raises TypeError: if the data is not of type NumericDataset
        :raises ValueError: if the model shape does not match data shape
        :return: None
        """

        batch_size = batch_size if batch_size is not None else self._batch_size
        epochs = epochs if epochs is not None else self._epochs
        tuner = kt.RandomSearch(
            lambda hp: self._build_automodel(data.shape[1:], hp=hp),  # Pass the additional input here
            objective=kt.Objective("loss", direction="min"),
            max_trials=10,
            directory=".",
            project_name='AutoTabularVAE_tuning',
            overwrite=True
        )
        tuner.search(data, epochs=epochs, batch_size=batch_size)
        self._model = tuner.get_best_models(num_models=1)[0]
        score = self._model.evaluate(data, data)
        print(score)
        self.training_info = TrainingInfo(
            loss_fn="ELBO",
            train_loss=tuner.get_best_models(num_models=1)[0].history["loss"][-1].numpy().item(),
            train_samples=data.shape[0],
            validation_loss=-1,
            validation_samples=0,
        )


    @classmethod
    def self_describe(cls):
        return ModelInfo(
            name=f"{cls.__module__}.{cls.__qualname__}",
            default_loss_function="ELBO LOSS",
            description="A Variational Autoencoder for data generation with automatic architecture selection",
            allowed_data=[
                AllowedData("float32", False),
                AllowedData("int32", False),
                AllowedData("int64", False),
                AllowedData("int32", True),
                AllowedData("str", True),
            ],
        ).get_model_info()
