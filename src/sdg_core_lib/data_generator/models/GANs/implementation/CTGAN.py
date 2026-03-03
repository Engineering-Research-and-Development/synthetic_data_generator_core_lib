import os

from sdg_core_lib.data_generator.models.UnspecializedModel import UnspecializedModel
from sdg_core_lib.data_generator.models.ModelInfo import ModelInfo
from sdg_core_lib.commons import AllowedData, DataType

os.environ["KERAS_BACKEND"] = "tensorflow"
from sdg_core_lib.data_generator.models.GANs.CTGANComponents import (
    CTGANGenerator,
    CTGANCritic,
    CTGANModel,
)
import keras
import tensorflow as tf
from sdg_core_lib.data_generator.models.TrainingInfo import TrainingInfo
import numpy as np


class CTGAN(UnspecializedModel):
    def __init__(
        self,
        metadata: list[dict],
        model_name: str,
        input_shape: str = None,
        load_path: str = None,
        gen_hidden=256,
        critic_hidden=256,
        pac_size=10,
        learning_rate=1e-3,
        batch_size=100,
        epochs=10,
        gen_steps=4,
        critic_dropout=0.2,
    ):
        super().__init__(
            self._clean_skeleton(metadata), model_name, input_shape, load_path
        )
        self._batch_size = batch_size
        self._epochs = epochs
        self._gen_steps = gen_steps
        self._pac_size = pac_size
        self._gen_hidden = gen_hidden
        self._critic_hidden = critic_hidden
        self._learning_rate = learning_rate
        self._critic_dropout = critic_dropout
        self._instantiate()

    @staticmethod
    def _clean_skeleton(skeleton):
        if skeleton != [{}]:
            return [
                item
                for item in skeleton
                if item["feature_type"] in ["continuous", "categorical"]
            ]
        return skeleton

    def infer_data_structure(self):
        cats, modes, idxs = [], [], []
        true_index = 0
        for col in self._metadata:
            try:
                f_size = int(col["feature_size"])
                if col["feature_type"] == "categorical":
                    cats.append(f_size)
                    # These are the actual global column indices in the train_tensor
                    idxs.extend(range(true_index, true_index + f_size))
                elif f_size <= 1:
                    raise AttributeError(
                        "Continous column after normalization must have at least size 2 (1 column "
                        "for the norm values and another for indicating the onehot of"
                        "a single mode"
                    )
                else:
                    modes.append(f_size - 1)
            except KeyError as e:
                raise AttributeError(
                    f"The CTGAN needs a valid data schema for each column, "
                    f"key {e.args[0]} is missing"
                )
            true_index += f_size
        if not cats:
            raise AttributeError("At least a categorical column must be passed!")
        return cats, modes, idxs

    def _build(self, input_shape: tuple[int, ...]):
        """
            This method is called during init if there is no load path,
            otherwise the method _load will be called
        :param input_shape:
        :return:
        """

        if (
            not isinstance(self._metadata, list)
            or not self._metadata
            or any(not isinstance(item, dict) or not item for item in self._metadata)
        ):
            raise AttributeError("CTGAN needs a data schema in order to work!")
        # Infer dimensions and indices
        (
            categories_per_discrete_column,
            modes_per_continuous_column,
            onehot_discrete_indexes,
        ) = self.infer_data_structure()
        self.generator = CTGANGenerator(
            self._metadata,
            modes_per_continuous_column,
            categories_per_discrete_column,
            self._gen_hidden,
        )
        self.critic = CTGANCritic(
            self._pac_size, self._critic_hidden, self._critic_dropout
        )
        return CTGANModel(self.generator, self.critic, onehot_discrete_indexes)

    def _load(self, folder_path: str):
        # Should set the _model variable CTGAN Model complete with Generator and Critic
        # Does NOT return the model
        # self._metadata is available
        _, _, onehot_discrete_indexes = self.infer_data_structure()
        critic = keras.saving.load_model(os.path.join(folder_path, "critic.keras"))
        generator = keras.saving.load_model(
            os.path.join(folder_path, "generator.keras")
        )
        self._model = CTGANModel(generator, critic, onehot_discrete_indexes)

        # Load probability_mass_function_list if it exists
        pmf_path = os.path.join(folder_path, "probability_mass_function_list.npz")
        if os.path.exists(pmf_path):
            pmf_data = np.load(pmf_path)
            # Convert back to list of TensorFlow tensors
            pmf_list = []
            for key in sorted(pmf_data.keys()):
                pmf_list.append(tf.convert_to_tensor(pmf_data[key], dtype=tf.float32))
            self._model.probability_mass_function_list = pmf_list
        # Also check for old .npy format for backward compatibility
        elif os.path.exists(
            os.path.join(folder_path, "probability_mass_function_list.npy")
        ):
            pmf_list = np.load(
                os.path.join(folder_path, "probability_mass_function_list.npy"),
                allow_pickle=True,
            )
            # Convert to TensorFlow tensors if needed
            if (
                isinstance(pmf_list, list)
                and len(pmf_list) > 0
                and isinstance(pmf_list[0], np.ndarray)
            ):
                self._model.probability_mass_function_list = [
                    tf.convert_to_tensor(pmf, dtype=tf.float32) for pmf in pmf_list
                ]
            else:
                self._model.probability_mass_function_list = pmf_list

    def save(self, folder_path: str):
        keras.saving.save_model(
            self._model.generator, os.path.join(folder_path, "generator.keras")
        )
        keras.saving.save_model(
            self._model.critic, os.path.join(folder_path, "critic.keras")
        )

        if (
            hasattr(self._model, "probability_mass_function_list")
            and self._model.probability_mass_function_list is not None
        ):
            # Convert TensorFlow tensors to numpy arrays before saving
            pmf_list = [
                tensor.numpy() for tensor in self._model.probability_mass_function_list
            ]
            np.savez(
                os.path.join(folder_path, "probability_mass_function_list.npz"),
                *pmf_list,
            )

    def train(self, data: np.ndarray):
        """
        The idea is to condense training hyperparams here and call
        Since learning_rate and other hyperparams comes from user, it should be better defining
        generator optimizer and critic optimizer here and pass them through the model.fit method.
        self._model.fit(data, gen_opt, crit_opt, ....)
        :param data:
        :return: Nothing
        IMPORTANT: Here TrainingInfo should be defined. See KerasBaseVAE train method
        """
        self._model.compile(
            g_optimizer=keras.optimizers.Adam(
                self._learning_rate, beta_1=0.5, beta_2=0.9
            ),
            d_optimizer=keras.optimizers.Adam(
                self._learning_rate, beta_1=0.5, beta_2=0.9
            ),
        )
        self._model._train_data = data
        probability_mass_function_list = self._model.get_pmfs(data)
        self._model.probability_mass_function_list = probability_mass_function_list
        history = self._model.fit(
            data, batch_size=self._batch_size, epochs=self._epochs, verbose=1
        )
        self.training_info = TrainingInfo(
            loss_fn="GeneratorLossWBCE",
            train_loss=history.history["g_loss"][-1].numpy().item(),
            train_samples=data.shape[0],
            validation_loss=-1,
            validation_samples=0,
        )

    def fine_tune(self, data: np.ndarray, **kwargs):
        raise NotImplementedError

    def infer(self, n_rows: int, **kwargs):
        return self._model.generate(n_rows)

    def set_hyperparameters(self, **kwargs):
        """
        Define some hyperarams that can be defined outside using kwargs
        :param kwargs:
        :return:
        """
        self._batch_size = int(kwargs.get("batch_size", self._batch_size))
        self._epochs = int(kwargs.get("epochs", self._epochs))
        self._pac_size = kwargs.get("pac_size", self._pac_size)
        self._gen_hidden = kwargs.get("gen_hidden", self._gen_hidden)
        self._critic_hidden = kwargs.get("critic_hidden", self._critic_hidden)
        self._learning_rate = float(kwargs.get("learning_rate", self._learning_rate))
        self._critic_dropout = kwargs.get("critic_dropout", self._critic_dropout)

    @classmethod
    def self_describe(cls):
        return ModelInfo(
            name=f"{cls.__module__}.{cls.__qualname__}",
            default_loss_function="Generator Adversary Loss with Log frequency weighted cross entropy",
            description="A Conditional Tabular Generative Adversarial Network for data generation",
            allowed_data=[
                AllowedData(DataType.float32, False),
                AllowedData(DataType.int32, False),
                AllowedData(DataType.int32, True),
                AllowedData(DataType.string, True),
            ],
        ).get_model_info()
