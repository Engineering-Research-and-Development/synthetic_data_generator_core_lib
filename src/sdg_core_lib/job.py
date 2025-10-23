import copy

from sdg_core_lib.dataset.Dataset import Dataset
from sdg_core_lib.dataset.TypedSubDataset import TypedSubDataset
from sdg_core_lib.dataset.ops.merge import merge_all_datasets
from sdg_core_lib.dataset.ops.split import split_into_subdataset
from sdg_core_lib.evaluate.TabularComparison import TabularComparisonEvaluator
from sdg_core_lib.data_generator.model_factory import model_factory
from sdg_core_lib.data_generator.models.UnspecializedModel import UnspecializedModel
from sdg_core_lib.process.processors import ProcessorRegistry


def job(
    model_info: dict, dataset: list, n_rows: int, save_filepath: str, train: bool
) -> tuple[list[dict], dict, UnspecializedModel, Dataset]:
    """
    Main function to run the job.

    This function will run the Synthetic Data Generation job. It will create an instance of the specified model or
    load the specified dataset, pre-process the data, train the model (if specified to do so), generate synthetic
    data, evaluate the generated data and save the results to the specified location.

    :param model_info: a dictionary containing the model's information
    :param dataset: a list of dataframes
    :param n_rows: the number of rows to generate
    :param save_filepath: the path to save the results
    :param train: a boolean indicating if the model should be trained
    :return: a tuple containing a list of metrics, a dictionary with the model's info, the trained model, and the generated dataset
    """

    if len(dataset) == 0:
        data_info = model_info.get("training_data_info", [])
        data = Dataset.from_json(data_info)
    else:
        data = Dataset.from_json(dataset)

    subdatasets = split_into_subdataset(data)
    model = None

    synthetic_subdatasets = []
    # TODO: Think for a way to handle mixed-dataset models
    for subdataset in subdatasets:
        input_shape = subdataset.get_processing_shape()
        model = model_factory(model_info, input_shape)
        pipeline_config = model.get_preprocessing_config()
        processor = ProcessorRegistry.get_processor(
            subdataset.data_type
        ).configure_and_setup(pipeline_config)

        if train:
            preprocessed_data, _ = processor.execute_preprocessing(
                subdataset.to_numpy(), None
            )
            model.train(data=preprocessed_data)
            model.save(save_filepath)
            processor.save_pipeline(save_filepath)

        processor.load_pipeline(save_filepath)
        predicted_data = model.infer(n_rows)
        postprocessed_data, _ = processor.execute_postprocessing(predicted_data, None)
        synthetic_subdataset = TypedSubDataset.from_data_and_metadata(
            postprocessed_data, subdataset.get_metadata()
        )
        synthetic_subdatasets.append(synthetic_subdataset)

    synthetic_data = merge_all_datasets(synthetic_subdatasets)

    return synthetic_data.to_json(), {"available": False}, model, synthetic_data
    report = {"available": False}

    if len(data.dataframe) > 0:
        evaluator = TabularComparisonEvaluator(
            real_data=data,
            synthetic_data=synthetic_data,
            numerical_columns=data.continuous_columns,
            categorical_columns=data.categorical_columns,
        )
        report = evaluator.compute()

    generated = copy.deepcopy(data)
    generated.dataframe = df_predict
    results = generated.parse_tabular_data_json()

    return results, report, model, data
