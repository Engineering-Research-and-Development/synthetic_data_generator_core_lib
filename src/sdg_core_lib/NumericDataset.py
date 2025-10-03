import numpy as np
import pandas as pd

NUMERICAL = "continuous"
CATEGORICAL = "categorical"
OTHER = "none"


class NumericDataset:
    """
    Class to handle numeric datasets.
    The class loads a dataset from a list of dictionaries into a pandas DataFrame.
    It also identifies which columns are numerical and which are categorical.
    """

    def __init__(self, dataset: list[dict]):
        self.dataset: list[dict] = dataset
        self.dataframe: pd.DataFrame = pd.DataFrame()
        self.columns: list[str] = []
        self.continuous_columns = []
        self.categorical_columns = []
        self.unrecognized_columns = []
        self.continuous_data: pd.DataFrame = pd.DataFrame()
        self.categorical_data: pd.DataFrame = pd.DataFrame()
        self.input_shape: str = ""
        self._configure()

    def _configure(self):
        """
        Convert data from requests into an easy-to-process dataframe
        dataset: [{
            column_data: [ ... ],
            column_name: str,
            column_type: str [continuous/categorical],
            column_datatype: str
        }]
        :return: a pandas Dataframe where each column is structured as expected
        :raises: ValueError if dataset is empty
        """
        data = self.dataset
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty")
        column_names = []
        categorical_columns = []
        numerical_columns = []
        unrecognized_columns = []
        data_structure = []
        for col in data:
            content = col.get("column_data", [])
            content_type = col.get("column_datatype", "object")
            column_name = col.get("column_name", "")
            column_type = col.get("column_type", "")
            data_structure.append(np.array(content, dtype=content_type))
            column_names.append(column_name)
            if column_type == NUMERICAL:
                numerical_columns.append(column_name)
            elif column_type == CATEGORICAL:
                categorical_columns.append(column_name)
            else:
                unrecognized_columns.append(column_name)

        input_data = {
            col["column_name"]: np.array(col.get("column_data", [])).tolist()
            for col in data
        }
        data_frame = pd.DataFrame(input_data)
        data_structure = np.array(data_frame.to_numpy().tolist())

        self.dataframe = data_frame
        self.columns = column_names
        self.categorical_columns = categorical_columns
        self.continuous_columns = numerical_columns
        self.unrecognized_columns = unrecognized_columns
        self.continuous_data = data_frame[numerical_columns]
        self.categorical_data = data_frame[categorical_columns]
        self.input_shape = str(data_structure.shape[1:])

    def _categorize_column(self, col):
        if col in self.continuous_columns:
            return NUMERICAL
        if col in self.categorical_columns:
            return CATEGORICAL
        return OTHER

    def parse_tabular_data_json(self) -> list[dict]:
        """
        Converts data from a dataframe into a list of dictionaries
        :return: a dictionary in form of:
        dataset: [{
            column_data: [ ... ],
            column_name: str,
            column_type: str [numerical/categorical],
            column_datatype: str
        }]
        """
        return [
            {
                "column_data": self.dataframe[col].to_numpy().tolist(),
                "column_name": col,
                "column_type": self._categorize_column(col),
                "column_datatype": str(self.dataframe[col].to_numpy().dtype),
            }
            for col in self.dataframe.columns
        ]

    def parse_data_to_registry(self) -> list[dict]:
        """
        Translates data structure from input coherence to a structured feature list
        :return:
        """
        feature_list = []
        for idx, col in enumerate(self.dataset):
            feat = {
                "feature_name": col.get("column_name", ""),
                "feature_position": idx,
                "is_categorical": (
                    True if col.get("column_type", "") == CATEGORICAL else False
                ),
                "type": col.get("column_datatype", ""),
            }
            feature_list.append(feat)
        return feature_list

    def get_data(self) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
        """
        Returns the data in the dataset as a tuple of 4 elements:

        1. The pandas DataFrame containing the data
        2. A list of column names
        3. A list of continuous column names
        4. A list of categorical column names

        :return: (dataframe, columns, continuous_columns, categorical_columns)
        :rtype: tuple[pandas.DataFrame, list[str], list[str], list[str]]
        """
        return (
            self.dataframe,
            self.columns,
            self.continuous_columns,
            self.categorical_columns,
        )

    @staticmethod
    def get_numpy_data(dataframe: pd.DataFrame) -> np.ndarray:
        """
        Correctly Returns numpy array with complex structures, like columns with type list
        :param dataframe: numpy dataframe
        :return: correctly structured numpy array
        """
        return np.array(dataframe.to_numpy().tolist())
