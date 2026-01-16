def check_min_max_boundary(min_value, max_value):
    """
    Ensures minimum and maximum values to be consistent in a interval
    :param min_value:
    :param max_value:
    :return:
    """
    if min_value > max_value:
        raise ValueError("min_value must be less than max_value")
