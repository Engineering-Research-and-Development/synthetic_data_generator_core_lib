from typing import Optional

from sdg_core_lib.data_generator.models.UnspecializedModel import UnspecializedModel
from sdg_core_lib.data_generator.models.keras.implementation.TabularVAE import (
    TabularVAE,
)
from sdg_core_lib.data_generator.models.keras.implementation.TimeSeriesVAE import (
    TimeSeriesVAE,
)
from sdg_core_lib.dataset.datasets import Dataset, Table, TimeSeries
from sdg_core_lib.post_process.FunctionApplier import FunctionApplier
from sdg_core_lib.preprocess.strategies.base_strategy import BasePreprocessingStrategy
from sdg_core_lib.preprocess.table_processor import TableProcessor
from sdg_core_lib.preprocess.base_processor import Processor
from sdg_core_lib.preprocess.strategies.vae_strategy import (
    TabularVAEPreprocessingStrategy,
    TimeSeriesVAEPreprocessingStrategy,
)
from sdg_core_lib.evaluate.tables import TabularComparisonEvaluator
from sdg_core_lib.evaluate.time_series import TimeSeriesComparisonEvaluator
import importlib
import os


def get_hyperparameters() -> dict:
    return {
        "epochs": os.environ.get("EPOCHS"),
        "learning_rate": os.environ.get("LEARNING_RATE"),
        "batch_size": os.environ.get("BATCH_SIZE"),
    }


class Job:
    dataset_mapping = {
        "table": {
            "dataset": Table,
            "evaluator": TabularComparisonEvaluator,
            "processor": TableProcessor,
        },
        "time_series": {
            "dataset": TimeSeries,
            "evaluator": TimeSeriesComparisonEvaluator,
            "processor": TableProcessor,
        },
    }

    model_strategy_mapping: dict[type, BasePreprocessingStrategy] = {
        TabularVAE: TabularVAEPreprocessingStrategy,
        TimeSeriesVAE: TimeSeriesVAEPreprocessingStrategy,
    }

    def __init__(
        self,
        n_rows: int,
        model_info: Optional[dict] = None,
        dataset: Optional[dict] = None,
        save_filepath: Optional[str] = None,
        functions: Optional[list[dict]] = None,
    ):
        self.model_info = model_info
        self.dataset = dataset
        self.n_rows = n_rows
        self.save_filepath = save_filepath
        self.functions = functions

    def _get_model_class(self) -> type:
        """
        Dynamically imports a class given its name.

        :return: the class itself
        """
        model_type = self.model_info.get("algorithm_name")
        module_name, class_name = model_type.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def _model_factory(
        self, preprocess_data: Dataset | None = None
    ) -> UnspecializedModel:
        model_file = self.model_info.get("image", None)
        model_name = self.model_info.get("model_name")
        input_shape = self.model_info.get("input_shape", None)
        metadata = None

        if self.dataset is not None:
            if input_shape is None:
                input_shape = preprocess_data.get_shape_for_model()
            metadata = preprocess_data.to_skeleton()

        model_class = self._get_model_class()
        model = model_class(
            metadata=metadata,
            model_name=model_name,
            input_shape=input_shape,
            load_path=model_file,
        )
        return model

    def _get_processor(self, dataset_type: str, model_class: type) -> Processor:
        processor_class = self.dataset_mapping[dataset_type]["processor"]
        processor = processor_class(self.save_filepath)

        strategy = self.model_strategy_mapping.get(
            model_class, BasePreprocessingStrategy
        )()
        processor.set_strategy(strategy)

        return processor

    def train(self) -> tuple[list[dict], dict, UnspecializedModel, list[dict]]:
        """
        Runs a pre-defined training job.

        This function will run the Synthetic Data Generation job. It will create an instance of the specified model or
        load the specified dataset, pre-process the data, train the model (if specified to do so), generate synthetic
        data, evaluate the generated data and save the results to the specified location.


        :return: a tuple containing a list of metrics, a dictionary with the model's info, the trained model, and the generated dataset
        """

        data_payload = self.dataset.get("data", [])
        dataset_type = self.dataset.get("dataset_type", "")
        dataset_class = self.dataset_mapping[dataset_type]["dataset"]
        model_class = self._get_model_class()
        dataset_evaluator_class = self.dataset_mapping[dataset_type]["evaluator"]

        data = dataset_class.from_json(data_payload)
        processor = self._get_processor(dataset_type, model_class)
        preprocessed_data = data.preprocess(processor)
        preprocess_schema = preprocessed_data.to_skeleton()

        model = self._model_factory(preprocessed_data)
        model.set_hyperparameters(**get_hyperparameters())
        model.train(data=preprocessed_data.get_computing_data())
        model.save(self.save_filepath)

        predicted_data = model.infer(self.n_rows)
        synthetic_data = preprocessed_data.clone(predicted_data)
        synthetic_data = synthetic_data.postprocess(processor)

        evaluator = dataset_evaluator_class(
            real_data=data,
            synthetic_data=synthetic_data,
        )
        report = evaluator.compute()
        results = synthetic_data.to_json()

        return results, report, model, preprocess_schema

    def infer(self) -> tuple[list[dict], dict]:
        dataset_type = self.dataset.get("dataset_type", "")
        dataset_class = self.dataset_mapping[dataset_type]["dataset"]
        dataset_evaluator_class = self.dataset_mapping[dataset_type]["evaluator"]

        data_payload = self.dataset.get("data", [])
        model_class = self._get_model_class()
        processor = self._get_processor(dataset_type, model_class)
        data = None
        if len(data_payload) == 0:
            data_skeleton = self.model_info.get("training_data_info")
            preprocessed_data = dataset_class.from_skeleton(data_skeleton)
        else:
            data = dataset_class.from_json(data_payload)
            preprocessed_data = data.preprocess(processor)

        model = self._model_factory(preprocessed_data)
        predicted_data = model.infer(self.n_rows)
        synthetic_data = preprocessed_data.clone(predicted_data)
        synthetic_data = synthetic_data.postprocess(processor)

        report = {"available": "false"}
        if data is not None:
            evaluator = dataset_evaluator_class(
                real_data=data,
                synthetic_data=synthetic_data,
            )
            report = evaluator.compute()

        results = synthetic_data.to_json()

        return results, report

    def generate_from_functions(self, dataset: Optional[Dataset] = None):
        """
        Generate a dataset from a list of functions.
        :param n_rows: number of rows to generate
        :param dataset: a Dataset object
        :return: a dataset in json format
        """
        from_scratch = False
        if dataset is None:
            from_scratch = True
        function_generator = FunctionApplier(
            self.functions, self.n_rows, from_scratch=from_scratch
        )
        dataset = function_generator.apply_all(dataset)
        return dataset.to_json()
