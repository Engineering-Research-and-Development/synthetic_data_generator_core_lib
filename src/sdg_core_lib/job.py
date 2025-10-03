import copy
import pandas as pd

from sdg_core_lib.evaluate.TabularComparison import TabularComparisonEvaluator
from sdg_core_lib.NumericDataset import NumericDataset
from sdg_core_lib.data_generator.model_factory import model_factory
from sdg_core_lib.data_generator.models.UnspecializedModel import UnspecializedModel


def job(
    model_info: dict, dataset: list, n_rows: int, save_filepath: str, train: bool
) -> tuple[list[dict], dict, UnspecializedModel, NumericDataset]:
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
        data = NumericDataset(dataset=data_info)
    else:
        data = NumericDataset(dataset=dataset)

    model = model_factory(model_info, data.input_shape)
    if train:
        model.train(data=data)
        model.save(save_filepath)

    predicted_data = model.infer(n_rows)
    df_predict = pd.DataFrame(data=predicted_data.tolist(), columns=data.columns)

    report = {"available": False}
    if len(data.dataframe) > 0:
        evaluator = TabularComparisonEvaluator(
            real_data=data.dataframe,
            synthetic_data=df_predict,
            numerical_columns=data.continuous_columns,
            categorical_columns=data.categorical_columns,
        )
        report = evaluator.compute()

    generated = copy.deepcopy(data)
    generated.dataframe = df_predict
    results = generated.parse_tabular_data_json()

    return results, report, model, data
