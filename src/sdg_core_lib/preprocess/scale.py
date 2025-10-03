from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np


def standardize_simple_tabular_input(
    train_data: np.array, test_data: np.array = None
) -> tuple[StandardScaler, np.array, np.array]:
    """
    Standardizes the tabular input data by scaling features to have zero mean and unit variance.

    :param train_data: A numpy array of shape (batch, features) representing the training data.
    :param test_data: An optional numpy array of shape (batch, features) representing the test data.
    :return: A tuple containing the fitted StandardScaler, the standardized training data, and the standardized test data
             if provided.
    :raises DataException: If the input data does not have the expected shape.
    """

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    if test_data is not None:
        test_data = scaler.transform(test_data)

    return scaler, train_data, test_data


def standardize_simple_tabular_time_series(
    train_data: np.array, test_data: np.array = None
) -> tuple[MinMaxScaler, np.array, np.array]:
    """
    Standardizes the time series data by scaling features to have zero mean and unit variance.

    :param train_data: A numpy array of shape (batch, features, steps) representing the training data.
    :param test_data: An optional numpy array of shape (batch, features, steps) representing the test data.
    :return: A tuple containing the fitted StandardScaler, the standardized training data, and the standardized test data
             if provided.
    :raises DataException: If the input data does not have the expected shape.
    """
    scaler = MinMaxScaler()

    batch, features, steps = train_data.shape

    x_reshaped = train_data.transpose(0, 2, 1).reshape(-1, features)
    x_scaled = scaler.fit_transform(x_reshaped)
    train_data = x_scaled.reshape(batch, steps, features).transpose(0, 2, 1)

    if test_data is not None:
        t_reshaped = test_data.transpose(0, 2, 1).reshape(-1, features)
        t_scaled = scaler.transform(t_reshaped)
        test_data = t_scaled.reshape(batch, steps, features).transpose(0, 2, 1)

    return scaler, train_data, test_data
