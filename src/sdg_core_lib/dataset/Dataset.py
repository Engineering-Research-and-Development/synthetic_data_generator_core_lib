from abc import ABC, abstractmethod


class Dataset(ABC):

    @classmethod
    @abstractmethod
    def from_json(cls):
        raise NotImplementedError

    @abstractmethod
    def to_json(self):
        raise NotImplementedError

    @abstractmethod
    def to_registry(self):
        raise NotImplementedError



