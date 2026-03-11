# Developer Documentation

## Overview

This comprehensive developer documentation covers the architecture, design patterns, and development practices for GENESIS Core Lib. Whether you're contributing to the project or extending it for your own use, this guide provides the technical details you need.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Design Patterns](#design-patterns)
4. [Development Setup](#development-setup)
5. [Contributing Guidelines](#contributing-guidelines)
6. [Testing Framework](#testing-framework)
7. [Code Organization](#code-organization)
8. [API Design](#api-design)
9. [Performance Optimization](#performance-optimization)
10. [Security Considerations](#security-considerations)

## Architecture Overview

### High-Level Architecture

GENESIS Core Lib follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│                   (User-facing API)                         │
├─────────────────────────────────────────────────────────────┤
│                    Service Layer                            │
│              (Job orchestration & logic)                   │
├─────────────────────────────────────────────────────────────┤
│                    Domain Layer                             │
│            (Models, Datasets, Functions)                   │
├─────────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                      │
│              (Storage, Logging, Utilities)                 │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│     Job     │───▶│   Dataset   │───▶│ Preprocess  │
└─────────────┘    └─────────────┘    └─────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Model    │◀───│   Functions │◀───│ Postprocess │
└─────────────┘    └─────────────┘    └─────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Storage   │    │ Evaluation  │    │   Logging   │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Key Design Principles

1. **Modularity**: Each component has a single responsibility
2. **Extensibility**: Easy to add new models, functions, and data types
3. **Testability**: All components are unit testable
4. **Performance**: Optimized for speed and memory usage
5. **Maintainability**: Clean, well-documented code

## Core Components

### Job System

The `Job` class is the main orchestrator that coordinates all components.

#### Class Structure
```python
class Job:
    def __init__(self, n_rows, model_info=None, dataset=None, save_filepath=None, functions=None):
        self.__model_info = model_info
        self.__dataset = dataset
        self.__n_rows = n_rows
        self.__save_filepath = save_filepath
        self.__functions = functions
        
    def train(self) -> tuple[list[dict], dict, UnspecializedModel, list[dict]]:
        # Main training pipeline
        
    def infer(self) -> tuple[list[dict], dict]:
        # Inference-only pipeline
        
    def generate_from_functions(self, dataset=None) -> list[dict]:
        # Function-based generation
```

#### Key Methods

##### _get_dataset_mapping()
Maps dataset types to appropriate handlers:
```python
@staticmethod
def _get_dataset_mapping(dataset_type: str) -> Type[DatasetMapping]:
    if dataset_type == "table":
        return TableMapping
    if dataset_type == "time_series":
        return TimeSeriesMapping
    return DatasetMapping
```

##### _model_factory()
Creates model instances dynamically:
```python
def _model_factory(self, preprocess_data: Dataset | None = None) -> UnspecializedModel:
    model_class = self._get_model_class()
    model = model_class(
        metadata=metadata,
        model_name=model_name,
        input_shape=input_shape,
        load_path=model_file,
    )
    return model
```

### Dataset Abstraction

The dataset system provides a unified interface for different data types.

#### Base Dataset Class
```python
class Dataset(ABC):
    @classmethod
    @abstractmethod
    def from_json(cls, json_data: list[dict]) -> "Dataset":
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def from_skeleton(cls, skeleton: list[dict]):
        raise NotImplementedError
    
    @abstractmethod
    def clone(self, new_data: np.ndarray) -> "Dataset":
        raise NotImplementedError
    
    @abstractmethod
    def get_computing_data(self) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def preprocess(self, processor: Processor) -> "Dataset":
        raise NotImplementedError
    
    @abstractmethod
    def postprocess(self, processor: Processor) -> "Dataset":
        raise NotImplementedError
```

#### Concrete Implementations

##### Table Dataset
```python
class Table(Dataset):
    def __init__(self, data: np.ndarray, columns: list[Column]):
        self.data = data
        self.columns = columns
        self.processor = None
    
    def from_json(cls, json_data: list[dict]) -> "Table":
        # Convert JSON to structured table
        pass
    
    def preprocess(self, processor: Processor) -> "Table":
        # Apply preprocessing transformations
        pass
```

### Model System

The model system provides an extensible framework for ML models.

#### UnspecializedModel Base Class
```python
class UnspecializedModel(ABC):
    def __init__(self, metadata, model_name, input_shape=None, load_path=None):
        self._metadata = metadata
        self.model_name = model_name
        self.input_shape = input_shape
        self._load_path = load_path
        self._model = None
    
    @abstractmethod
    def _build(self, input_shape: tuple[int, ...]):
        raise NotImplementedError
    
    @abstractmethod
    def train(self, data: np.ndarray):
        raise NotImplementedError
    
    @abstractmethod
    def infer(self, n_rows: int, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def save(self, folder_path):
        raise NotImplementedError
```

#### Model Implementations

##### GAN Models
```python
class TabularGAN(UnspecializedModel):
    def _build(self, input_shape):
        # Build generator and discriminator
        self.generator = self._build_generator(input_shape)
        self.discriminator = self._build_discriminator(input_shape)
        self.gan = self._build_gan()
    
    def train(self, data):
        # Implement GAN training logic
        pass
```

##### VAE Models
```python
class TabularVAE(UnspecializedModel):
    def _build(self, input_shape):
        # Build encoder and decoder
        self.encoder = self._build_encoder(input_shape)
        self.decoder = self._build_decoder(input_shape)
        self.vae = self._build_vae()
    
    def train(self, data):
        # Implement VAE training logic
        pass
```

### Function System

The function system provides a flexible way to generate and modify data.

#### UnspecializedFunction Base Class
```python
class UnspecializedFunction(ABC):
    parameters = []  # List of Parameter objects
    description = ""
    priority = Priority.MEDIUM
    is_generative = False
    allowed_data = []  # List of allowed data types
    
    def __init__(self, parameters: list[Parameter]):
        self.parameters = parameters
        self._check_parameters()
    
    @abstractmethod
    def apply(self, n_rows: int, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        raise NotImplementedError
```

#### Function Categories

##### Generation Functions
Create data from scratch:
```python
class LinearFunction(UnspecializedFunction):
    parameters = [
        Parameter("m", "1.0", "float"),
        Parameter("q", "0.0", "float"),
        Parameter("min_value", "0.0", "float"),
        Parameter("max_value", "1.0", "float"),
    ]
    description = "Generates linear data following y=mx+q"
    priority = Priority.MAX
    is_generative = True
    
    def apply(self, n_rows: int, data: np.ndarray):
        x = np.linspace(self.min_value, self.max_value, n_rows)
        y = self.m * x + self.q
        return y.reshape(-1, 1), np.empty((n_rows, 1)), True
```

##### Modification Functions
Transform existing data:
```python
class NoiseFunction(UnspecializedFunction):
    parameters = [
        Parameter("noise_level", "0.1", "float"),
        Parameter("distribution", "normal", "string"),
    ]
    is_generative = False
    
    def apply(self, n_rows: int, data: np.ndarray):
        noise = np.random.normal(0, self.noise_level, data.shape)
        modified_data = data + noise
        return modified_data, data, False
```

### Processing System

The processing system handles data transformation and normalization.

#### Processor Base Class
```python
class Processor(ABC):
    def __init__(self, save_filepath: str = None):
        self.save_filepath = save_filepath
        self.strategy = None
    
    def set_strategy(self, strategy):
        self.strategy = strategy
    
    @abstractmethod
    def preprocess(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError
    
    @abstractmethod
    def postprocess(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError
```

#### Strategy Pattern Implementation

##### Table Processing Strategy
```python
class TableProcessingStrategy(ProcessingStrategy):
    def preprocess(self, dataset: Table) -> Table:
        # Handle numeric columns
        numeric_columns = dataset.get_numeric_columns()
        for col in numeric_columns:
            data = dataset.get_column_data(col)
            normalized_data = self.normalize_numeric(data)
            dataset.set_column_data(col, normalized_data)
        
        # Handle categorical columns
        categorical_columns = dataset.get_categorical_columns()
        for col in categorical_columns:
            data = dataset.get_column_data(col)
            encoded_data = self.encode_categorical(data)
            dataset.set_column_data(col, encoded_data)
        
        return dataset
```

### Evaluation System

The evaluation system assesses the quality of synthetic data.

#### Evaluator Base Class
```python
class Evaluator(ABC):
    def __init__(self, real_data: Dataset, synthetic_data: Dataset):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
    
    @abstractmethod
    def compute(self) -> dict:
        raise NotImplementedError
```

#### Evaluation Metrics

##### Statistical Similarity
```python
class StatisticalSimilarityMetric:
    def compute(self, real_data, synthetic_data):
        # Compare distributions using KS test
        ks_statistic, p_value = ks_2samp(real_data, synthetic_data)
        
        # Compare means and variances
        mean_diff = np.abs(np.mean(real_data) - np.mean(synthetic_data))
        var_diff = np.abs(np.var(real_data) - np.var(synthetic_data))
        
        return {
            "ks_statistic": ks_statistic,
            "p_value": p_value,
            "mean_difference": mean_diff,
            "variance_difference": var_diff
        }
```

##### Correlation Preservation
```python
class CorrelationPreservationMetric:
    def compute(self, real_data, synthetic_data):
        real_corr = np.corrcoef(real_data.T)
        synthetic_corr = np.corrcoef(synthetic_data.T)
        
        correlation_diff = np.abs(real_corr - synthetic_corr)
        mean_correlation_diff = np.mean(correlation_diff)
        
        return {
            "correlation_difference": mean_correlation_diff,
            "correlation_matrix": synthetic_corr
        }
```

## Design Patterns

### Strategy Pattern

Used extensively for processing strategies and model strategies.

#### Implementation
```python
class ProcessingStrategy(ABC):
    @abstractmethod
    def preprocess(self, dataset):
        pass
    
    @abstractmethod
    def postprocess(self, dataset):
        pass

class TableProcessingStrategy(ProcessingStrategy):
    def preprocess(self, dataset):
        # Table-specific preprocessing
        pass

class TimeSeriesProcessingStrategy(ProcessingStrategy):
    def preprocess(self, dataset):
        # Time series-specific preprocessing
        pass
```

### Factory Pattern

Used for creating models and functions dynamically.

#### Model Factory
```python
class ModelFactory:
    @staticmethod
    def create_model(model_config: dict) -> UnspecializedModel:
        model_type = model_config["algorithm_name"]
        module_name, class_name = model_type.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        return model_class(**model_config)
```

#### Function Factory
```python
def function_factory(function_config: dict) -> UnspecializedFunction:
    function_name = function_config["function_name"]
    parameters = function_config.get("parameters", {})
    
    # Map function names to classes
    function_map = {
        "LinearFunction": LinearFunction,
        "QuadraticFunction": QuadraticFunction,
        "SinusoidalFunction": SinusoidalFunction,
    }
    
    function_class = function_map.get(function_name)
    if function_class is None:
        raise ValueError(f"Unknown function: {function_name}")
    
    # Convert parameters to Parameter objects
    param_objects = [
        Parameter(name, str(value), infer_type(value))
        for name, value in parameters.items()
    ]
    
    return function_class(param_objects)
```

### Observer Pattern

Used for monitoring training progress and logging.

#### Implementation
```python
class TrainingObserver(ABC):
    @abstractmethod
    def on_epoch_start(self, epoch: int):
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: dict):
        pass
    
    @abstractmethod
    def on_training_complete(self, final_metrics: dict):
        pass

class ProgressLogger(TrainingObserver):
    def on_epoch_end(self, epoch: int, metrics: dict):
        logger.info(f"Epoch {epoch}: {metrics}")
```

### Template Method Pattern

Used in the base classes to define algorithm structure.

#### Implementation
```python
class UnspecializedModel(ABC):
    def train(self, data: np.ndarray):
        # Template method defining training structure
        self._prepare_data(data)
        self._initialize_model()
        self._train_loop()
        self._finalize_training()
    
    def _prepare_data(self, data):
        # Common data preparation
        pass
    
    @abstractmethod
    def _initialize_model(self):
        # Model-specific initialization
        pass
    
    def _train_loop(self):
        # Common training loop
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
    
    @abstractmethod
    def _train_epoch(self, epoch: int):
        # Model-specific epoch training
        pass
```

### Command Pattern

Used for function application and job execution.

#### Implementation
```python
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass

class GenerateDataCommand(Command):
    def __init__(self, job: Job):
        self.job = job
    
    def execute(self):
        return self.job.train()

class ApplyFunctionCommand(Command):
    def __init__(self, function: UnspecializedFunction, data: np.ndarray):
        self.function = function
        self.data = data
    
    def execute(self):
        return self.function.apply(len(self.data), self.data)
```

## Development Setup

### Prerequisites

- Python 3.12+
- Git
- Development tools (pytest, black, flake8, mypy)

### Environment Setup

#### Clone Repository
```bash
git clone https://github.com/emiliocimino/generator_core_lib.git
cd generator_core_lib
```

#### Create Development Environment
```bash
# Using UV (recommended)
uv sync --dev

# Or using pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

#### Pre-commit Setup
```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit manually
pre-commit run --all-files
```

### Development Tools

#### Code Formatting
```bash
# Format code with black
black src/ tests/

# Check formatting
black --check src/ tests/
```

#### Linting
```bash
# Run flake8
flake8 src/ tests/

# Run mypy for type checking
mypy src/
```

#### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sdg_core_lib --cov-report=html

# Run specific test file
pytest tests/test_job.py

# Run with verbose output
pytest -v
```

## Contributing Guidelines

### Code Style

Follow these style guidelines:

#### Python Style
- Use Black for code formatting
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Keep lines under 88 characters (Black default)

#### Documentation Style
- Use docstrings for all public methods and classes
- Follow Google docstring format
- Include type hints in docstrings
- Provide examples for complex functions

#### Example Docstring
```python
def train_model(data: np.ndarray, epochs: int = 100) -> dict:
    """Train a synthetic data generation model.
    
    Args:
        data: Training data as numpy array
        epochs: Number of training epochs
        
    Returns:
        Dictionary containing training metrics
        
    Raises:
        ValueError: If data is empty or invalid
        
    Example:
        >>> data = np.random.normal(0, 1, (1000, 10))
        >>> metrics = train_model(data, epochs=50)
        >>> print(metrics['loss'])
    """
    pass
```

### Pull Request Process

#### Before Submitting
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure they pass
5. Update documentation if needed
6. Submit a pull request

#### Pull Request Requirements
- Clear description of changes
- Tests for new functionality
- Documentation updates
- No breaking changes without justification
- All tests passing

#### Code Review Process
1. Automated checks must pass
2. At least one human review required
3. Address all review comments
4. Maintain backward compatibility

### Release Process

#### Version Management
- Use semantic versioning (MAJOR.MINOR.PATCH)
- Update version in pyproject.toml
- Create git tag for releases
- Update CHANGELOG.md

#### Release Checklist
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Version number updated
- [ ] CHANGELOG updated
- [ ] Tag created
- [ ] PyPI package published

## Testing Framework

### Test Structure

```
tests/
├── unit/                  # Unit tests
│   ├── test_job.py
│   ├── test_models.py
│   ├── test_functions.py
│   └── test_datasets.py
├── integration/           # Integration tests
│   ├── test_pipelines.py
│   └── test_workflows.py
├── performance/          # Performance tests
│   └── test_benchmarks.py
└── fixtures/             # Test data
    ├── sample_data.json
    └── model_configs.json
```

### Test Categories

#### Unit Tests
Test individual components in isolation:

```python
import pytest
from unittest.mock import Mock, patch
from sdg_core_lib.job import Job

class TestJob:
    def test_job_initialization(self):
        job = Job(n_rows=100)
        assert job._Job__n_rows == 100
    
    def test_get_dataset_mapping_table(self):
        mapping = Job._get_dataset_mapping("table")
        assert mapping.__name__ == "TableMapping"
    
    @patch('sdg_core_lib.job.importlib.import_module')
    def test_get_model_class(self, mock_import):
        mock_module = Mock()
        mock_class = Mock()
        mock_module.TabularGAN = mock_class
        mock_import.return_value = mock_module
        
        job = Job(n_rows=100, model_info={
            "algorithm_name": "module.TabularGAN"
        })
        
        result = job._get_model_class()
        assert result == mock_class
```

#### Integration Tests
Test component interactions:

```python
import pytest
from sdg_core_lib import Job

class TestIntegration:
    def test_full_pipeline(self):
        dataset_config = {
            "dataset_type": "table",
            "data": [{"x": 1, "y": 2}, {"x": 2, "y": 4}]
        }
        
        model_config = {
            "algorithm_name": "sdg_core_lib.data_generator.models.GANs.TabularGAN",
            "model_name": "test_model"
        }
        
        job = Job(
            n_rows=10,
            model_info=model_config,
            dataset=dataset_config
        )
        
        results, metrics, model, schema = job.train()
        
        assert len(results) == 10
        assert isinstance(metrics, dict)
        assert model is not None
        assert isinstance(schema, list)
```

#### Performance Tests
Test performance characteristics:

```python
import pytest
import time
from sdg_core_lib import Job

class TestPerformance:
    def test_training_speed(self):
        start_time = time.time()
        
        # Run training
        job = Job(n_rows=1000, model_info=model_config, dataset=dataset_config)
        job.train()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Should complete within reasonable time
        assert training_time < 300  # 5 minutes max
    
    def test_memory_usage(self):
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run memory-intensive operation
        job = Job(n_rows=10000, model_info=model_config, dataset=dataset_config)
        job.train()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 1024 * 1024 * 1024  # 1GB max
```

### Test Utilities

#### Fixtures
```python
import pytest
import numpy as np

@pytest.fixture
def sample_data():
    return [
        {"x": 1.0, "y": 2.0},
        {"x": 2.0, "y": 4.0},
        {"x": 3.0, "y": 6.0}
    ]

@pytest.fixture
def model_config():
    return {
        "algorithm_name": "sdg_core_lib.data_generator.models.GANs.TabularGAN",
        "model_name": "test_model"
    }

@pytest.fixture
def job_instance(model_config, sample_data):
    return Job(
        n_rows=100,
        model_info=model_config,
        dataset={"dataset_type": "table", "data": sample_data}
    )
```

#### Mock Objects
```python
from unittest.mock import Mock

@pytest.fixture
def mock_model():
    model = Mock()
    model.train.return_value = None
    model.infer.return_value = np.random.normal(0, 1, (100, 2))
    return model
```

### Test Data Management

#### Test Data Generation
```python
def generate_test_data(n_rows: int, n_features: int) -> list[dict]:
    """Generate test data for unit tests."""
    data = []
    for i in range(n_rows):
        row = {}
        for j in range(n_features):
            row[f"feature_{j}"] = np.random.normal(0, 1)
        data.append(row)
    return data
```

#### Test Data Storage
```python
import json
import tempfile

def save_test_data(data: list[dict]) -> str:
    """Save test data to temporary file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        return f.name
```

## Code Organization

### Directory Structure

```
src/sdg_core_lib/
├── __init__.py
├── job.py                    # Main orchestration
├── commons.py               # Shared constants and utilities
├── mappings.py             # Type mappings and configurations
├── browser.py               # Data browsing utilities
├── data_generator/          # ML models
│   ├── __init__.py
│   └── models/
│       ├── __init__.py
│       ├── UnspecializedModel.py
│       ├── ModelInfo.py
│       ├── TrainingInfo.py
│       ├── GANs/
│       └── VAEs/
├── dataset/                 # Data abstractions
│   ├── __init__.py
│   ├── datasets.py
│   └── columns.py
├── post_process/            # Data processing and functions
│   ├── __init__.py
│   ├── FunctionApplier.py
│   ├── function_factory.py
│   ├── function_utils.py
│   └── functions/
│       ├── __init__.py
│       ├── UnspecializedFunction.py
│       ├── Parameter.py
│       └── implementation/
├── preprocess/              # Data preprocessing
│   ├── __init__.py
│   ├── base_processor.py
│   └── table_processor.py
└── evaluate/                # Quality evaluation
    ├── __init__.py
    ├── base_evaluator.py
    └── metrics/
```

### Module Responsibilities

#### Core Modules
- `job.py`: Main orchestration and user interface
- `commons.py`: Shared constants, enums, and utilities
- `mappings.py`: Type mappings and configuration classes

#### Data Generator
- `models/`: ML model implementations
- `GANs/`: GAN-specific models
- `VAEs/`: VAE-specific models

#### Dataset
- `datasets.py`: Dataset abstractions and base classes
- `columns.py`: Column type definitions and handling

#### Post Process
- `FunctionApplier.py`: Function application logic
- `functions/`: Data generation and modification functions

#### Preprocess
- `base_processor.py`: Abstract processor interface
- `table_processor.py`: Table-specific processing

#### Evaluate
- `base_evaluator.py`: Evaluation framework
- `metrics/`: Quality metric implementations

### Import Organization

#### Import Style
```python
# Standard library imports
import os
import json
from typing import Optional, Dict, List

# Third-party imports
import numpy as np
import pandas as pd
from tensorflow import keras

# Local imports
from sdg_core_lib.commons import DataType, AllowedData
from sdg_core_lib.dataset.datasets import Dataset
from sdg_core_lib.data_generator.models.UnspecializedModel import UnspecializedModel
```

#### Circular Import Prevention
- Use forward references for type hints
- Import interfaces rather than implementations when possible
- Use dependency injection to reduce coupling

```python
# Use forward reference to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sdg_core_lib.dataset.datasets import Dataset

class Model:
    def process_data(self, data: 'Dataset') -> 'Dataset':
        pass
```

## API Design

### Design Principles

#### Consistency
- Use consistent naming conventions
- Follow similar patterns across APIs
- Maintain backward compatibility

#### Simplicity
- Minimal, focused interfaces
- Sensible defaults
- Clear error messages

#### Extensibility
- Abstract base classes for extension
- Plugin architecture for functions
- Configuration-driven behavior

### Public API

#### Main Entry Points
```python
# Primary user interface
from sdg_core_lib import Job

# Secondary interfaces
from sdg_core_lib.data_generator.models import UnspecializedModel
from sdg_core_lib.post_process.functions import UnspecializedFunction
from sdg_core_lib.dataset.datasets import Dataset
```

#### Configuration API
```python
# Model configuration
model_config = {
    "algorithm_name": "path.to.ModelClass",
    "model_name": "unique_identifier",
    "hyperparameters": {...}
}

# Dataset configuration
dataset_config = {
    "dataset_type": "table|time_series",
    "data": [...],
    "preprocessing": {...}
}

# Function configuration
function_config = {
    "function_name": "FunctionName",
    "parameters": {...}
}
```

#### Error Handling
```python
# Custom exceptions
class GenesisCoreLibError(Exception):
    """Base exception for GENESIS Core Lib."""
    pass

class ConfigurationError(GenesisCoreLibError):
    """Raised when configuration is invalid."""
    pass

class ModelTrainingError(GenesisCoreLibError):
    """Raised when model training fails."""
    pass

class DataValidationError(GenesisCoreLibError):
    """Raised when data validation fails."""
    pass
```

### Internal API

#### Abstract Interfaces
```python
# Model interface
class UnspecializedModel(ABC):
    @abstractmethod
    def train(self, data: np.ndarray):
        pass
    
    @abstractmethod
    def infer(self, n_rows: int):
        pass

# Dataset interface
class Dataset(ABC):
    @abstractmethod
    def preprocess(self, processor: Processor) -> "Dataset":
        pass
    
    @abstractmethod
    def postprocess(self, processor: Processor) -> "Dataset":
        pass
```

#### Plugin Architecture
```python
# Function registry
class FunctionRegistry:
    _functions = {}
    
    @classmethod
    def register(cls, name: str, function_class: type):
        cls._functions[name] = function_class
    
    @classmethod
    def get_function(cls, name: str) -> type:
        return cls._functions.get(name)

# Registration decorator
def register_function(name: str):
    def decorator(cls):
        FunctionRegistry.register(name, cls)
        return cls
    return decorator

@register_function("CustomFunction")
class CustomFunction(UnspecializedFunction):
    pass
```

## Performance Optimization

### Memory Management

#### Efficient Data Handling
```python
# Use generators for large datasets
def data_generator(data: list[dict], batch_size: int = 1000):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Memory-efficient preprocessing
def preprocess_in_batches(dataset: Dataset, batch_size: int = 1000):
    processed_batches = []
    for batch in data_generator(dataset.data, batch_size):
        processed_batch = preprocess_batch(batch)
        processed_batches.append(processed_batch)
    return concatenate_batches(processed_batches)
```

#### Memory Monitoring
```python
import psutil
import gc

def monitor_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "rss": memory_info.rss,  # Physical memory
        "vms": memory_info.vms,  # Virtual memory
        "percent": process.memory_percent()
    }

def cleanup_memory():
    gc.collect()
    # Clear TensorFlow session if needed
    tf.keras.backend.clear_session()
```

### Computational Optimization

#### Vectorization
```python
# Use numpy vectorization instead of loops
def vectorized_computation(data: np.ndarray) -> np.ndarray:
    # Good: Vectorized operation
    return np.exp(data) / (1 + np.exp(data))
    
    # Bad: Loop-based operation
    # result = []
    # for x in data:
    #     result.append(np.exp(x) / (1 + np.exp(x)))
    # return np.array(result)
```

#### Parallel Processing
```python
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

def parallel_function_application(functions: list, data: np.ndarray):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(func.apply, len(data), data)
            for func in functions
        ]
        results = [future.result() for future in futures]
    return results
```

### GPU Optimization

#### GPU Memory Management
```python
import tensorflow as tf

def configure_gpu():
    # Enable memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def use_mixed_precision():
    # Enable mixed precision training
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
```

#### Batch Size Optimization
```python
def optimize_batch_size(model, data):
    # Find optimal batch size based on available memory
    batch_sizes = [16, 32, 64, 128, 256]
    
    for batch_size in batch_sizes:
        try:
            # Test if batch size fits in memory
            test_batch = data[:batch_size]
            model.train_on_batch(test_batch)
            return batch_size
        except tf.errors.ResourceExhaustedError:
            continue
    
    return 16  # Fallback to small batch size
```

### Caching Strategy

#### Model Caching
```python
import pickle
from pathlib import Path

class ModelCache:
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, model_config: dict) -> str:
        import hashlib
        config_str = json.dumps(model_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def load_model(self, model_config: dict):
        cache_key = self.get_cache_key(model_config)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_model(self, model, model_config: dict):
        cache_key = self.get_cache_key(model_config)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(model, f)
```

## Security Considerations

### Input Validation

#### Data Validation
```python
def validate_input_data(data: list[dict]) -> bool:
    if not isinstance(data, list):
        raise ValueError("Data must be a list")
    
    if len(data) == 0:
        raise ValueError("Data cannot be empty")
    
    if not all(isinstance(row, dict) for row in data):
        raise ValueError("All rows must be dictionaries")
    
    # Validate schema consistency
    first_row_keys = set(data[0].keys())
    for i, row in enumerate(data[1:], 1):
        if set(row.keys()) != first_row_keys:
            raise ValueError(f"Row {i} has different schema than first row")
    
    return True
```

#### Configuration Validation
```python
def validate_model_config(config: dict) -> bool:
    required_fields = ["algorithm_name", "model_name"]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate algorithm name format
    algorithm_name = config["algorithm_name"]
    if not isinstance(algorithm_name, str) or "." not in algorithm_name:
        raise ValueError("algorithm_name must be in format 'module.ClassName'")
    
    return True
```

### Secure Model Loading

#### Safe Model Import
```python
import importlib.util
import sys

def safe_import(module_path: str, class_name: str):
    # Validate module path
    if not module_path.startswith("sdg_core_lib"):
        raise ImportError("Only modules within sdg_core_lib can be imported")
    
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        
        # Validate that it's a proper class
        if not isinstance(cls, type):
            raise ValueError(f"{class_name} is not a class")
        
        return cls
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import {module_path}.{class_name}: {e}")
```

#### Model File Validation
```python
import magic

def validate_model_file(file_path: str) -> bool:
    # Check file type
    file_type = magic.from_file(file_path, mime=True)
    allowed_types = ['application/octet-stream', 'application/x-hdf']
    
    if file_type not in allowed_types:
        raise ValueError(f"Invalid model file type: {file_type}")
    
    # Check file size
    file_size = os.path.getsize(file_path)
    max_size = 1024 * 1024 * 1024  # 1GB
    
    if file_size > max_size:
        raise ValueError(f"Model file too large: {file_size} bytes")
    
    return True
```

### Privacy Protection

#### Differential Privacy
```python
import numpy as np

class DifferentialPrivacyWrapper:
    def __init__(self, epsilon: float = 1.0, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.sensitivity = sensitivity
    
    def add_noise(self, data: np.ndarray) -> np.ndarray:
        scale = self.sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, data.shape)
        return data + noise
    
    def private_query(self, data: np.ndarray, query_func) -> np.ndarray:
        result = query_func(data)
        return self.add_noise(result)
```

#### Data Anonymization
```python
import hashlib
import re

def anonymize_data(data: list[dict], sensitive_fields: list) -> list[dict]:
    anonymized = []
    
    for row in data:
        new_row = row.copy()
        
        for field in sensitive_fields:
            if field in new_row:
                # Hash sensitive fields
                value = str(new_row[field])
                hashed = hashlib.sha256(value.encode()).hexdigest()
                new_row[field] = f"hashed_{hashed[:8]}"
        
        anonymized.append(new_row)
    
    return anonymized
```

### Access Control

#### Permission System
```python
from enum import Enum
from typing import Set

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    TRAIN = "train"
    GENERATE = "generate"

class User:
    def __init__(self, user_id: str, permissions: Set[Permission]):
        self.user_id = user_id
        self.permissions = permissions
    
    def has_permission(self, permission: Permission) -> bool:
        return permission in self.permissions

class AccessControl:
    def __init__(self):
        self.users = {}
    
    def add_user(self, user: User):
        self.users[user.user_id] = user
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        user = self.users.get(user_id)
        if not user:
            return False
        return user.has_permission(permission)
```

#### Audit Logging
```python
import logging
from datetime import datetime

class AuditLogger:
    def __init__(self, log_file: str = "audit.log"):
        self.logger = logging.getLogger("audit")
        self.logger.addHandler(logging.FileHandler(log_file))
        self.logger.setLevel(logging.INFO)
    
    def log_access(self, user_id: str, action: str, resource: str):
        timestamp = datetime.now().isoformat()
        log_entry = f"{timestamp} - User {user_id} performed {action} on {resource}"
        self.logger.info(log_entry)
    
    def log_model_training(self, user_id: str, model_name: str, data_size: int):
        self.log_access(user_id, "TRAIN", f"model:{model_name},size:{data_size}")
    
    def log_data_generation(self, user_id: str, n_rows: int, model_name: str):
        self.log_access(user_id, "GENERATE", f"rows:{n_rows},model:{model_name}")
```

This comprehensive developer documentation provides all the technical details needed to understand, extend, and contribute to GENESIS Core Lib effectively.
