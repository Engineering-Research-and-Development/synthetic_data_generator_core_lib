import numpy as np
from abc import ABC


class Column(ABC):
    def __init__(self, name: str, value_type: str, position: int, values: np.ndarray):
        self.name = name
        self.value_type = value_type
        self.position = position
        self.values = values
        self.internal_shape = self.get_internal_shape()
        self.column_type = None

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




class NumericColumn(Column):
    def __init__(self, name: str, value_type: str, position: int, values: np.ndarray):
        super().__init__(name, value_type, position, values)
        self.column_type = "continuous"

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        metadata.update({"column_type": "continuous"})
        return metadata

    def get_boundaries(self) -> tuple[float, float]:
        return self.values.min(), self.values.max()

    def to_categorical(self, n_bins: int) -> 'CategoricalColumn':
        bins = np.linspace(self.values.min(), self.values.max(), n_bins)
        binned_values = np.digitize(self.values, bins)
        return CategoricalColumn(self.name, self.value_type, self.position, binned_values)



class CategoricalColumn(Column):
    def __init__(self, name: str, value_type: str, position: int, values: np.ndarray):
        super().__init__(name, value_type, position, values)
        self.column_type = "categorical"

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        metadata.update({"column_type": "categorical"})
        return metadata

    def get_categories(self) -> list[str]:
        return list(set(self.values))


class PrimaryKeyColumn(Column):
    def __init__(self, name: str, value_type: str, position: int, values: np.ndarray):
        super().__init__(name, value_type, position, values)
        self.column_type = "primary_key"

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        metadata.update({"column_type": "primary_key"})
        return metadata

