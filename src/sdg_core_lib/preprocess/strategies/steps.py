from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
import os
import skops.io as sio
from sklearn.mixture import BayesianGaussianMixture


class Step(ABC):
    def __init__(self, type_name: str, position: int, col_name: str, mode: str):
        self.type_name = type_name
        self.mode = mode
        self.position = position
        self.col_name = col_name
        self.operator = None
        self.filename = (
            f"{self.position}_{self.col_name}_{self.mode}_{self.type_name}.skops"
        )

    @abstractmethod
    def _set_operator(self):
        raise NotImplementedError

    def save_if_not_exist(self, directory_path: str):
        if self.operator is None:
            raise ValueError("Operator is not created")
        os.makedirs(directory_path, exist_ok=True)
        filename = os.path.join(directory_path, self.filename)
        if not os.path.exists(filename):
            sio.dump(self.operator, filename)

    def load(self, directory_path: str):
        filename = os.path.join(directory_path, self.filename)
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Operator file not found: {filename}")
        self.operator = sio.load(filename)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.operator = self._set_operator()
        return self.operator.fit_transform(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.operator is None:
            raise ValueError("Operator not initialized")
        return self.operator.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.operator is None:
            raise ValueError("Operator not initialized")
        return self.operator.inverse_transform(data)


class NoneStep(Step):
    def __init__(self, position: int, mode="", type_name="none"):
        super().__init__(type_name=type_name, position=position, col_name="", mode=mode)

    def save_if_not_exist(self, directory_path: str):
        pass

    def load(self, directory_path: str):
        pass

    def _set_operator(self):
        pass

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data


class ScalerWrapper(Step):
    def __init__(
        self,
        position: int,
        col_name: str,
        mode: Literal["minmax", "standard"] = "standard",
        type_name="scaler",
    ):
        super().__init__(
            type_name=type_name, position=position, col_name=col_name, mode=mode
        )

    def _set_operator(self):
        if self.mode == "minmax":
            return MinMaxScaler()
        elif self.mode == "standard":
            return StandardScaler()
        else:
            raise ValueError("Invalid mode while setting the scaler")


class OrdinalEncoderWrapper(Step):
    def __init__(
        self, position: int, col_name: str, mode="ordinal", type_name="encoder"
    ):
        super().__init__(
            type_name=type_name, position=position, col_name=col_name, mode=mode
        )

    def _set_operator(self):
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)


class OneHotEncoderWrapper(Step):
    def __init__(
        self, position: int, col_name: str, mode="one_hot", type_name="encoder"
    ):
        super().__init__(
            type_name=type_name, position=position, col_name=col_name, mode=mode
        )

    def _set_operator(self):
        return OneHotEncoder(handle_unknown="error")

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return super().fit_transform(data).toarray()

    def transform(self, data: np.ndarray) -> np.ndarray:
        return super().transform(data).toarray()

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        # Numerical stability for all zeros
        data = data + np.ones(data.shape) * 1e-9
        return super().inverse_transform(data)


class PerModeNormalization(Step):
    """
    This step estimates using variational gaussian mixtures models
    the number of modes the data may come from and performs mode specific
    normalization that will be later used by a CTGAN. This step also
    saves this information in order to perform inverse transformations
    """

    def __init__(
        self,
        position: int,
        col_name: str,
        mode: str = "",
        type_name="per_mode_normalization",
        n_components=10,
        max_iter=1000,
        random_state=42,
    ):
        super().__init__(
            type_name=type_name, position=position, col_name=col_name, mode=mode
        )
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state

    def _set_operator(self):
        vbgmm = BayesianGaussianMixture(
            n_components=self.n_components,
            weight_concentration_prior_type="dirichlet_process",
            covariance_type="full",
            max_iter=1000,
            random_state=self.random_state,
        )
        return vbgmm

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.operator = self._set_operator()
        self.operator.fit(data)
        return self.transform(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.operator is None:
            raise ValueError("Operator not initialized")
        column = data.reshape(-1, 1)
        active_weights_indx = np.where(self.operator.weights_ > 0.01)
        weights = self.operator.weights_[active_weights_indx]
        means = self.operator.means_[active_weights_indx].flatten()
        stds = np.sqrt(self.operator.covariances_[active_weights_indx].flatten())
        mixture_probability_density = []
        for w, m, s in zip(weights, means, stds):
            mixture_probability_density.append(
                w
                * PerModeNormalization._gaussian_probability_density_function(
                    column, m, s
                )
            )
        marginal_mixture_probability_density = np.hstack(mixture_probability_density)
        responsibilities = PerModeNormalization._compute_responsibilities(
            marginal_mixture_probability_density
        )
        rng = np.random.default_rng(self.random_state)
        n, K = responsibilities.shape
        sampled_mode = np.array(
            [rng.choice(K, p=responsibilities[i]) for i in range(n)]
        )
        f = np.zeros((n, K), dtype=int)
        f[np.arange(n), sampled_mode] = 1
        mu_sel = means[sampled_mode]
        std_sel = stds[sampled_mode]
        normalized_value = (column.reshape(-1) - mu_sel) / (4.0 * std_sel)
        to_return = np.concatenate([normalized_value.reshape(-1, 1), f], axis=1)
        return to_return

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.operator is None:
            raise ValueError("Operator not initialized")
        active_weights_indx = np.where(self.operator.weights_ > 0.01)
        means = self.operator.means_[active_weights_indx].flatten()
        stds = np.sqrt(self.operator.covariances_[active_weights_indx].flatten())

        # Handle both 1D and 2D input data
        if data.ndim == 1:
            data = data.reshape(1, -1)

        active_modes = np.argmax(data[:, 1:], axis=1)

        # Get the means and stds for the active modes
        selected_mus = means[active_modes]
        selected_devs = stds[active_modes]

        # Get the normalized values (first column)
        normalized_values = data[:, 0]

        # Denormalize the values
        values = (normalized_values * 4 * selected_devs) + selected_mus

        # Always return 2D array with shape (n_samples, 1) for consistency
        return values.reshape(-1, 1)

    @staticmethod
    def _gaussian_probability_density_function(
        x: np.ndarray, mean: np.ndarray, std: np.ndarray
    ):
        """
        This function computes the probability density function of the gaussian mixture
        given the mean and standard deviation
        """
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(
            -0.5 * (x - mean) ** 2 / (std**2)
        )

    @staticmethod
    def _compute_responsibilities(pdf_per_mode: np.ndarray) -> np.ndarray:
        return pdf_per_mode / pdf_per_mode.sum(axis=1, keepdims=True)
