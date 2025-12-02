import numpy as np


class Column:
    def __init__(
        self,
        name: str,
        value_type: str,
        position: int,
        values: np.ndarray,
        column_type: str,
    ):
        self.name = name
        self.value_type = value_type
        self.position = position
        self.values = values
        self.column_type = column_type
        self.internal_shape = self.get_internal_shape()

    def get_internal_shape(self) -> tuple[int, ...]:
        return self.values.shape

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "value_type": self.value_type,
            "position": self.position,
            "internal_shape": self.internal_shape,
        }

    def get_data(self) -> np.ndarray:
        return self.values


class Numeric(Column):
    def __init__(
        self,
        name: str,
        value_type: str,
        position: int,
        values: np.ndarray,
        column_type: str,
    ):
        super().__init__(name, value_type, position, values, column_type)

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        metadata.update({"column_type": "continuous"})
        return metadata

    def get_boundaries(self) -> tuple[float, float]:
        return self.values.min(), self.values.max()

    def to_categorical(self, n_bins: int = 10) -> "Categorical":
        bins = np.linspace(self.values.min(), self.values.max(), n_bins)
        binned_values = np.digitize(self.values, bins)
        return Categorical(
            self.name, self.value_type, self.position, binned_values, "categorical"
        )


class Categorical(Column):
    def __init__(
        self,
        name: str,
        value_type: str,
        position: int,
        values: np.ndarray,
        column_type: str,
    ):
        super().__init__(name, value_type, position, values, column_type)

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        metadata.update({"column_type": "categorical"})
        return metadata

    def get_categories(self) -> list[str]:
        return list(set(self.values))
