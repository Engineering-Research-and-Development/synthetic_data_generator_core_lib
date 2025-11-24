from sdg_core_lib.dataset.datasets import Dataset, Table, TimeSeries
from sdg_core_lib.evaluate.TabularComparison import TabularComparisonEvaluator
from sdg_core_lib.data_generator.model_factory import model_factory
from sdg_core_lib.data_generator.models.UnspecializedModel import UnspecializedModel

dataset_mapping: dict[str, type[Dataset]] = {
    "table": Table,
    "time_series": TimeSeries
}

def train(
    model_info: dict, dataset: dict, n_rows: int, save_filepath: str
) -> tuple[list[dict], dict, UnspecializedModel, list[dict]]:
    """
    Main function to run the job.

    This function will run the Synthetic Data Generation job. It will create an instance of the specified model or
    load the specified dataset, pre-process the data, train the model (if specified to do so), generate synthetic
    data, evaluate the generated data and save the results to the specified location.

    :param model_info: a dictionary containing the model's information
    :param dataset: a list of dataframes
    :param n_rows: the number of rows to generate
    :param save_filepath: the path to save the results
    :return: a tuple containing a list of metrics, a dictionary with the model's info, the trained model, and the generated dataset
    """

    data_payload = dataset["data"]
    dataset_type = dataset["dataset_type"]
    data = dataset_mapping[dataset_type].from_json(data_payload, save_filepath)

    preprocessed_data = data.preprocess()
    preprocess_schema = data.to_skeleton()
    model = model_factory(model_info, preprocessed_data.get_shape_for_model())
    model.train(data=preprocessed_data.get_computing_data())
    model.save(save_filepath)

    predicted_data = model.infer(n_rows)
    synthetic_data = preprocessed_data.clone(predicted_data)
    synthetic_data = synthetic_data.postprocess()

    evaluator = TabularComparisonEvaluator(
        real_data=data,
        synthetic_data=synthetic_data,

    )
    report = evaluator.compute()
    results = synthetic_data.to_json()

    return results, report, model, preprocess_schema


def infer(
    model_info: dict, dataset: dict, n_rows: int, save_filepath: str
) -> tuple[list[dict], dict, UnspecializedModel, Dataset]:
    pass
