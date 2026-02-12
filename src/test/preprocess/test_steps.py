import shutil
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sdg_core_lib.preprocess.strategies.steps import (
    ScalerWrapper,
    NoneStep,
    OrdinalEncoderWrapper,
    OneHotEncoderWrapper,
    PerModeNormalization,
)


@pytest.fixture
def temp_dir():
    """Create and clean up a temporary directory for test files."""
    test_dir = Path("test_temp_dir")
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    shutil.rmtree(test_dir, ignore_errors=True)


class TestScalerWrapper:
    """Test suite for ScalerWrapper class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample test data."""
        return np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)

    @pytest.mark.parametrize(
        "mode,expected_scaler_type",
        [("standard", StandardScaler), ("minmax", MinMaxScaler)],
    )
    def test_initialization(self, mode, expected_scaler_type):
        """Test that ScalerWrapper initializes with correct parameters."""
        step = ScalerWrapper(position=0, col_name="test_col", mode=mode)
        assert step.type_name == "scaler"
        assert step.mode == mode
        assert step.position == 0
        assert step.col_name == "test_col"
        assert step.filename == f"0_test_col_{mode}_scaler.skops"

    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        step = ScalerWrapper(position=0, col_name="test_col", mode="standard")
        transformed = step.fit_transform(sample_data)
        assert transformed.shape == sample_data.shape
        assert step.operator is not None
        assert hasattr(step.operator, "transform")

    def test_inverse_transform(self, sample_data):
        """Test inverse_transform method."""
        step = ScalerWrapper(position=0, col_name="test_col", mode="minmax")
        transformed = step.fit_transform(sample_data)
        inverse_transformed = step.inverse_transform(transformed)
        np.testing.assert_allclose(inverse_transformed, sample_data, rtol=1e-6)

    def test_save_and_load(self, sample_data, temp_dir):
        """Test save and load functionality."""
        step = ScalerWrapper(position=0, col_name="test_col", mode="standard")
        step.fit_transform(sample_data)
        save_path = temp_dir / "test_scaler"
        step.save_if_not_exist(str(save_path))
        assert (save_path / f"{step.filename}").exists()
        loaded_step = ScalerWrapper(position=0, col_name="test_col", mode="standard")
        loaded_step.load(str(save_path))
        assert isinstance(loaded_step.operator, StandardScaler)
        assert loaded_step.position == 0
        assert loaded_step.col_name == "test_col"


class TestNoneStep:
    """Test suite for NoneStep class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample test data."""
        return np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)

    def test_initialization(self):
        """Test that NoneStep initializes with correct parameters."""
        step = NoneStep(position=1)
        assert step.type_name == "none"
        assert step.position == 1
        assert step.col_name == ""

    def test_fit_transform_returns_same_data(self, sample_data):
        """Test that fit_transform returns the input data unchanged."""
        step = NoneStep(position=0)
        result = step.fit_transform(sample_data)
        np.testing.assert_array_equal(result, sample_data)

    def test_transform_returns_same_data(self, sample_data):
        """Test that transform returns the input data unchanged."""
        step = NoneStep(position=0)
        result = step.transform(sample_data)
        np.testing.assert_array_equal(result, sample_data)

    def test_inverse_transform_returns_same_data(self, sample_data):
        """Test that inverse_transform returns the input data unchanged."""
        step = NoneStep(position=0)
        result = step.inverse_transform(sample_data)
        np.testing.assert_array_equal(result, sample_data)


class TestLabelEncoderWrapper:
    """Test suite for LabelEncoderWrapper class."""

    @pytest.fixture
    def categorical_data(self):
        """Generate sample categorical data."""
        return np.array(["a", "b", "c", "a", "b"]).reshape(-1, 1)

    def test_fit_transform(self, categorical_data):
        """Test fit_transform with categorical data."""
        step = OrdinalEncoderWrapper(position=0, col_name="category")
        transformed = step.fit_transform(categorical_data)
        assert transformed.shape == categorical_data.shape  # Same number of samples
        assert set(transformed.flatten().tolist()) == {
            0,
            1,
            2,
        }  # Should encode to 0, 1, 2

    def test_inverse_transform(self, categorical_data):
        """Test inverse_transform to get back original categories."""
        step = OrdinalEncoderWrapper(position=0, col_name="category")
        transformed = step.fit_transform(categorical_data)
        inverse_transformed = step.inverse_transform(transformed)
        np.testing.assert_array_equal(inverse_transformed, categorical_data)


class TestOneHotEncoderWrapper:
    """Test suite for OneHotEncoderWrapper class."""

    @pytest.fixture
    def categorical_data(self):
        """Generate sample categorical data."""
        return np.array(["a", "b", "c", "a", "b"]).reshape(-1, 1)

    def test_fit_transform(self, categorical_data):
        """Test fit_transform with one-hot encoding."""
        step = OneHotEncoderWrapper(position=0, col_name="category")
        transformed = step.fit_transform(categorical_data)
        assert transformed.shape == (5, 3)  # 5 samples, 3 categories
        assert np.all(transformed.sum(axis=1) == 1)  # Each row sums to 1 (one-hot)
        assert set(transformed.flatten().tolist()) == {0.0, 1.0}  # Only 0s and 1s

    def test_inverse_transform(self, categorical_data):
        """Test inverse_transform to get back original categories."""
        step = OneHotEncoderWrapper(position=0, col_name="category")
        transformed = step.fit_transform(categorical_data)
        inverse_transformed = step.inverse_transform(transformed)
        np.testing.assert_array_equal(inverse_transformed, categorical_data)

    def test_handles_unknown_categories(self):
        """Test behavior with unknown categories during transform."""
        train_data = np.array(["a", "b", "c"]).reshape(-1, 1)
        test_data = np.array(["a", "d", "b"]).reshape(-1, 1)  # 'd' is unknown
        step = OneHotEncoderWrapper(position=0, col_name="category")
        step.fit_transform(train_data)
        with pytest.raises(ValueError):
            step.transform(test_data)


class TestPerModeNormalization:
    """Test suite for PerModeNormalization class."""

    @pytest.fixture
    def multimodal_data(self):
        """Generate sample multimodal data."""
        # Create data with multiple modes (clusters)
        np.random.seed(42)
        mode1 = np.random.normal(0, 1, 50)  # First mode
        mode2 = np.random.normal(5, 1.5, 30)  # Second mode
        mode3 = np.random.normal(-3, 0.8, 20)  # Third mode
        return np.concatenate([mode1, mode2, mode3]).reshape(-1, 1)

    @pytest.fixture
    def simple_data(self):
        """Generate simple unimodal data."""
        np.random.seed(42)
        return np.random.normal(0, 1, 100).reshape(-1, 1)

    def test_initialization(self):
        """Test that PerModeNormalization initializes with correct parameters."""
        step = PerModeNormalization(position=0, col_name="test_col")
        assert step.type_name == "per_mode_normalization"
        assert step.position == 0
        assert step.col_name == "test_col"
        assert step.n_components == 10
        assert step.max_iter == 1000
        assert step.random_state == 42
        assert step.filename == "0_test_col__per_mode_normalization.skops"

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        step = PerModeNormalization(
            position=1,
            col_name="custom_col",
            n_components=5,
            max_iter=500,
            random_state=123,
        )
        assert step.n_components == 5
        assert step.max_iter == 500
        assert step.random_state == 123

    def test_fit_transform_multimodal_data(self, multimodal_data):
        """Test fit_transform with multimodal data."""
        step = PerModeNormalization(position=0, col_name="test_col")
        transformed = step.fit_transform(multimodal_data)

        # Output should have normalized values + one-hot encoded modes
        assert (
            transformed.shape[0] == multimodal_data.shape[0]
        )  # Same number of samples
        assert (
            transformed.shape[1] > 1
        )  # Should have multiple columns (normalized + modes)

        # First column should be normalized values (roughly between -3 and 3)
        normalized_values = transformed[:, 0]
        assert np.all(
            np.abs(normalized_values) < 10
        )  # Reasonable range for normalized data

        # Mode columns should be one-hot encoded (each row should sum to 1 for mode columns)
        mode_columns = transformed[:, 1:]
        assert np.allclose(mode_columns.sum(axis=1), 1.0)  # Each row sums to 1

    def test_fit_transform_simple_data(self, simple_data):
        """Test fit_transform with simple unimodal data."""
        step = PerModeNormalization(position=0, col_name="test_col")
        transformed = step.fit_transform(simple_data)

        assert transformed.shape[0] == simple_data.shape[0]
        assert transformed.shape[1] > 1

        # Should still work with unimodal data
        mode_columns = transformed[:, 1:]
        assert np.allclose(mode_columns.sum(axis=1), 1.0)

    def test_transform_after_fit(self, multimodal_data):
        """Test transform method after fitting."""
        step = PerModeNormalization(position=0, col_name="test_col")
        step.fit_transform(multimodal_data)  # Fit the model

        # Test transform on new data
        new_data = np.array([[1.0], [2.0], [-1.0]])
        transformed = step.transform(new_data)

        assert transformed.shape[0] == new_data.shape[0]
        assert transformed.shape[1] > 1
        assert np.allclose(transformed[:, 1:].sum(axis=1), 1.0)

    def test_inverse_transform(self, multimodal_data):
        """Test inverse_transform to recover original data."""
        step = PerModeNormalization(position=0, col_name="test_col")
        transformed = step.fit_transform(multimodal_data)
        inverse_transformed = step.inverse_transform(transformed)

        # Should recover original shape
        assert inverse_transformed.shape == multimodal_data.shape

        # Values should be reasonably close (allowing for some approximation error)
        np.testing.assert_allclose(
            inverse_transformed.flatten(),
            multimodal_data.flatten(),
            rtol=0.1,  # Allow 10% relative tolerance due to mode assignment randomness
            atol=0.5,  # Allow 0.5 absolute tolerance
        )

    def test_inverse_transform_1d_input(self, multimodal_data):
        """Test inverse_transform with 1D input data."""
        step = PerModeNormalization(position=0, col_name="test_col")
        transformed = step.fit_transform(multimodal_data)

        # Test with single row
        single_row = transformed[0]
        result = step.inverse_transform(single_row)
        inverse_transformed = step.inverse_transform(transformed)
        assert result.shape == (1, 1)
        assert np.allclose(multimodal_data, inverse_transformed)

    def test_save_and_load(self, multimodal_data, temp_dir):
        """Test save and load functionality."""
        step = PerModeNormalization(position=0, col_name="test_col")
        step.fit_transform(multimodal_data)

        # Save the step
        save_path = temp_dir / "test_per_mode"
        step.save_if_not_exist(str(save_path))
        assert (save_path / step.filename).exists()

        # Load the step
        loaded_step = PerModeNormalization(position=0, col_name="test_col")
        loaded_step.load(str(save_path))

        assert loaded_step.operator is not None
        assert loaded_step.position == 0
        assert loaded_step.col_name == "test_col"

        # Test that loaded step produces similar results
        original_transformed = step.transform(multimodal_data[:10])
        loaded_transformed = loaded_step.transform(multimodal_data[:10])
        np.testing.assert_allclose(original_transformed, loaded_transformed)

    def test_transform_without_fit_raises_error(self, simple_data):
        """Test that transform without fitting raises an error."""
        step = PerModeNormalization(position=0, col_name="test_col")
        with pytest.raises(ValueError, match="Operator not initialized"):
            step.transform(simple_data)

    def test_inverse_transform_without_fit_raises_error(self, simple_data):
        """Test that inverse_transform without fitting raises an error."""
        step = PerModeNormalization(position=0, col_name="test_col")
        with pytest.raises(ValueError, match="Operator not initialized"):
            step.inverse_transform(simple_data)

    def test_static_gaussian_pdf_function(self):
        """Test the static Gaussian probability density function."""
        x = np.array([[0.0], [1.0], [-1.0]])
        mean = np.array([0.0])
        std = np.array([1.0])

        pdf = PerModeNormalization._gaussian_probability_density_function(x, mean, std)

        # PDF should be positive
        assert np.all(pdf > 0)
        # PDF at mean should be highest
        assert pdf[0] > pdf[1] and pdf[0] > pdf[2]

    def test_static_compute_responsibilities(self):
        """Test the static responsibilities computation."""
        # Create sample PDF values for 2 modes
        pdf_values = np.array(
            [
                [0.8, 0.2],  # Sample 1: higher probability for mode 0
                [0.3, 0.7],  # Sample 2: higher probability for mode 1
                [0.5, 0.5],  # Sample 3: equal probabilities
            ]
        )

        responsibilities = PerModeNormalization._compute_responsibilities(pdf_values)

        # Each row should sum to 1
        assert np.allclose(responsibilities.sum(axis=1), 1.0)
        # Values should be between 0 and 1
        assert np.all(responsibilities >= 0) and np.all(responsibilities <= 1)
