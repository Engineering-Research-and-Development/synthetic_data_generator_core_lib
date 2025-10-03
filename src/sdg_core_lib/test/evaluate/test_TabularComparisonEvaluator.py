import pytest
import pandas as pd

from sdg_core_lib.evaluate.TabularComparison import TabularComparisonEvaluator


@pytest.fixture()
def real_data():
    return pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}
    )


@pytest.fixture()
def synthetic_data():
    return pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}
    )


@pytest.fixture()
def evaluator_correct(real_data, synthetic_data):
    return TabularComparisonEvaluator(real_data, synthetic_data, ["a", "b"], ["c", "d"])


def test_init(evaluator_correct, real_data, synthetic_data):
    assert evaluator_correct._numerical_columns == ["a", "b"]
    assert evaluator_correct._categorical_columns == ["c", "d"]
    assert evaluator_correct._real_data.equals(real_data)
    assert evaluator_correct._synthetic_data.equals(synthetic_data)


def test_evaluate(evaluator_correct):
    report = evaluator_correct.compute()
    print(report)
    assert "statistical_metrics" in report
    assert "adherence_metrics" in report
    assert "novelty_metrics" in report
    statistical_metrics = report["statistical_metrics"]
    adherence_metrics = report["adherence_metrics"]
    novelty_metrics = report["novelty_metrics"]
    statistical_metrics_titles = [metric["title"] for metric in statistical_metrics]
    assert "Total Statistical Compliance" in statistical_metrics_titles
    assert "Categorical Features Cramer's V" in statistical_metrics_titles
    assert "Numerical Features Wasserstein Distance" in statistical_metrics_titles
    assert (
        len(adherence_metrics[0]["value"])
        == len(evaluator_correct._categorical_columns)
        and adherence_metrics[0]["title"]
        == "Synthetic Categories Adherence to Real Categories"
    )
    assert (
        len(adherence_metrics[1]["value"]) == len(evaluator_correct._numerical_columns)
        and adherence_metrics[1]["title"]
        == "Synthetic Numerical Min-Max Boundaries Adherence"
    )
    assert (
        0 <= novelty_metrics[0]["value"] <= 100
        and novelty_metrics[0]["title"] == "Unique Synthetic Data"
    )
    assert (
        0 <= novelty_metrics[1]["value"] <= 100
        and novelty_metrics[1]["title"] == "New Synthetic Data"
    )


def test_evaluate_cramer_v_distance(evaluator_correct):
    cramer_v = evaluator_correct._evaluate_cramer_v_distance()
    print(cramer_v)
    assert 0 <= cramer_v <= 1


def test_evaluate_wasserstein_distance(evaluator_correct):
    wass_distance = evaluator_correct._evaluate_wasserstein_distance()
    assert 0 <= wass_distance <= 1
