from sdg_core_lib.dataset.Dataset import Dataset

class FusedDataset:
    def __init__(self, datasets: list[Dataset]):
        self.index = datasets
