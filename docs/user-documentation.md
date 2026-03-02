# User Documentation

## Overview

This comprehensive user documentation covers all aspects of using GENESIS Core Lib, from basic concepts to advanced techniques. Whether you're a beginner or an experienced data scientist, this guide will help you master synthetic data generation.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Data Types](#data-types)
3. [Model Types](#model-types)
4. [Configuration](#configuration)
5. [API Reference](#api-reference)
6. [Examples and Use Cases](#examples-and-use-cases)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Core Concepts

### The Job System

The `Job` class is the central orchestrator in GENESIS Core Lib. It manages the entire synthetic data generation pipeline from data loading to model training and inference.

#### Job Lifecycle

```
Data Input → Preprocessing → Model Training → Data Generation → Postprocessing → Evaluation
```

#### Key Components

1. **Dataset Configuration**: Defines the input data and its type
2. **Model Configuration**: Specifies the ML model to use
3. **Functions**: Mathematical functions for data generation/modification
4. **Output Settings**: Where to save models and results

### Data Processing Pipeline

GENESIS Core Lib follows a structured approach to data processing:

#### 1. Data Input
- Load real data or define generation functions
- Validate data format and structure
- Extract metadata and schema information

#### 2. Preprocessing
- Normalize and scale features
- Encode categorical variables
- Handle missing values
- Prepare data for model training

#### 3. Model Training
- Initialize the specified ML model
- Train on preprocessed data
- Optimize hyperparameters
- Save trained model

#### 4. Data Generation
- Use trained model for inference
- Generate specified number of rows
- Apply post-processing functions

#### 5. Postprocessing
- Reverse preprocessing transformations
- Apply custom functions
- Format output data

#### 6. Evaluation
- Compare synthetic vs real data
- Calculate quality metrics
- Generate evaluation report

## Data Types

### Tabular Data

Tabular data is structured data organized in rows and columns, similar to a spreadsheet or database table.

#### Characteristics
- Fixed schema with defined columns
- Mixed data types (numeric, categorical, text)
- Independent observations
- Suitable for most business datasets

#### Configuration Example

```python
dataset_config = {
    "dataset_type": "table",
    "data": [
        {
            "column_data": [13.71, 13.4, 13.27, 13.17, 14.13],
            "column_name": "alcohol",
            "column_type": "continuous",
            "column_datatype": "float64"
        },
        {
            "column_data": [5.65, 3.91, 4.28, 2.59, 4.1],
            "column_name": "malic_acid",
            "column_type": "continuous",
            "column_datatype": "float64"
        }
    ]
}
```

#### Supported Column Types
- **Numeric**: Integers and floats
- **Categorical**: Discrete categories
- **Text**: String values
- **Boolean**: True/False values
- **DateTime**: Timestamps and dates

#### Best Practices
- Ensure consistent data types across columns
- Handle missing values appropriately
- Limit categorical cardinality (<100 categories recommended)
- Normalize numeric features when possible

### Time Series Data

Time series data consists of observations collected sequentially over time.

#### Characteristics
- Temporal ordering is significant
- May have trends, seasonality, and patterns
- Can be univariate or multivariate
- Requires special handling for temporal dependencies

#### Configuration Example

```python
dataset_config = {
    "dataset_type": "time_series",
    "data": [
        {
            "column_name": "experiment_id",
            "column_type": "group_index",
            "column_data": [1, 1, 1, 2, 2, 2, 3, 3],
            "column_datatype": "int"
        },
        {
            "column_name": "time",
            "column_type": "primary_key",
            "column_data": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "column_datatype": "int"
        },
        {
            "column_name": "value1",
            "column_type": "continuous",
            "column_data": [1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3],
            "column_datatype": "float"
        },
        {
            "column_name": "category",
            "column_type": "categorical",
            "column_data": ["a", "b", "c", "a", "b", "c", "a", "b", "c"],
            "column_datatype": "str"
        }
    ]
}
```

#### Required Columns
- **group_index**: Identifies different experiments/time series groups
- **primary_key**: Time index within each experiment
- **continuous**: Numeric measurement values
- **categorical**: Discrete category values

#### Best Practices
- Ensure consistent experiment lengths
- Provide proper group_index for experiment identification
- Use primary_key for temporal ordering
- Handle missing time points appropriately
- Ensure regular time intervals
- Consider seasonality and trends
- Use appropriate frequency settings

### Custom Data Types

GENESIS Core Lib supports custom data types through its extensible architecture.

#### Creating Custom Data Types

```python
from sdg_core_lib.dataset.datasets import Dataset

class CustomDataset(Dataset):
    @classmethod
    def from_json(cls, json_data):
        # Implementation for loading custom data
        pass
    
    def preprocess(self, processor):
        # Custom preprocessing logic
        pass
    
    def postprocess(self, processor):
        # Custom postprocessing logic
        pass
```

## Model Types

### VAEs (Variational Autoencoders)

VAEs learn a compressed latent representation of data and can generate new samples by sampling from the latent space.

#### When to Use VAEs
- When smooth interpolation is desired
- For structured latent space
- When training stability is a concern
- For datasets with clear patterns

#### Available VAE Models

##### TabularVAE
```python
model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TabularVAE.TabularVAE",
    "model_name": "tabular_vae_model"
}
```

**Features:**
- Stable training process
- Interpretable latent space
- Good for feature analysis
- Handles missing values well

##### TimeSeriesVAE
```python
model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TimeSeriesVAE.TimeSeriesVAE",
    "model_name": "time_series_vae_model"
}
```

**Features:**
- Captures temporal dependencies
- Handles variable-length sequences
- Preserves temporal patterns

#### VAE Best Practices
- Use appropriate latent dimensionality
- Monitor reconstruction loss
- Consider the beta-VAE variant for disentangled representations
- Validate latent space interpretability

### CTGAN (Conditional Tabular GAN)

CTGAN is a specialized GAN for tabular data that handles mixed data types effectively.

#### When to Use CTGAN
- Complex data distributions
- High-dimensional data
- When generating highly realistic data is critical
- Tabular data with complex feature interactions

#### Available CTGAN Models

##### CTGAN
```python
model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.GANs.implementation.CTGAN.CTGAN",
    "model_name": "ctgan_model"
}
```

**Features:**
- Handles mixed data types
- Preserves feature correlations
- Good for medium-sized datasets (1K-100K rows)

#### CTGAN Best Practices
- Use sufficient training data (1000+ rows recommended)
- Monitor training stability
- Adjust learning rates if training fails
- Consider data preprocessing for better convergence

### Model Selection Guide

| Dataset Size | Data Complexity | Recommended Model | Reason |
|-------------|-----------------|-------------------|---------|
| < 1,000 rows | Low | TabularVAE | More stable with small data |
| 1,000-10,000 rows | Medium | TabularVAE or CTGAN | Both work well |
| > 10,000 rows | High | CTGAN | Can capture complex distributions |
| Any size | Time series | TimeSeriesVAE | Specialized for temporal data |
| Any size | Very high dimensional | TabularVAE | More stable training |

## Configuration

### Dataset Configuration

#### Basic Structure
```python
dataset_config = {
    "dataset_type": "table|time_series|custom",
    "data": [list_of_columns],
    "metadata": {
        # Optional metadata
    }
}
```

#### Advanced Options
```python
dataset_config = {
    "dataset_type": "table",
    "data": data_payload,
    "preprocessing": {
        "normalize_numeric": True,
        "encode_categorical": True,
        "handle_missing": "mean|median|mode"
    },
    "validation": {
        "check_schema": True,
        "validate_types": True
    }
}
```

### Model Configuration

#### Basic Structure
```python
model_config = {
    "algorithm_name": "path.to.model.Class",
    "model_name": "unique_identifier",
    "input_shape": "auto|specific_shape",
    "image": "path/to/saved_model"  # For loading pre-trained models
}
```

#### Advanced Options
```python
model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TabularVAE.TabularVAE",
    "model_name": "advanced_vae",
    "hyperparameters": {
        "epochs": 1000,
        "learning_rate": 0.0002,
        "batch_size": 32,
        "latent_dim": 100
    },
    "architecture": {
        "hidden_layers": [256, 128, 64],
        "activation": "relu",
        "dropout": 0.2
    }
}
```

### Function Configuration

#### Generation Functions
```python
functions = [
    {
        "feature": "target_column",
        "function_name": "LinearFunction",
        "parameters": {
            "m": 1.0,
            "q": 0.0,
            "min_value": 0.0,
            "max_value": 1.0
        }
    }
]
```

#### Available Functions

##### LinearFunction
Generates linear data following y = mx + q
```python
{
    "function_name": "LinearFunction",
    "parameters": {
        "m": 1.0,          # slope
        "q": 0.0,          # y-intercept
        "min_value": 0.0,  # minimum x value
        "max_value": 1.0   # maximum x value
    }
}
```

##### QuadraticFunction
Generates quadratic data following y = ax² + bx + c
```python
{
    "function_name": "QuadraticFunction",
    "parameters": {
        "a": 1.0,          # quadratic coefficient
        "b": 0.0,          # linear coefficient
        "c": 0.0,          # constant term
        "min_value": 0.0,  # minimum x value
        "max_value": 1.0   # maximum x value
    }
}
```

##### SinusoidalFunction
Generates sinusoidal data
```python
{
    "function_name": "SinusoidalFunction",
    "parameters": {
        "amplitude": 1.0,  # wave amplitude
        "frequency": 1.0,  # wave frequency
        "phase": 0.0,      # phase shift
        "min_value": 0.0,  # minimum x value
        "max_value": 6.28  # maximum x value (2π)
    }
}
```

##### NormalDistributionSample
Generates data from normal distribution
```python
{
    "function_name": "NormalDistributionSample",
    "parameters": {
        "mean": 0.0,       # distribution mean
        "std_dev": 1.0,    # standard deviation
        "min_value": -3.0, # minimum sample value
        "max_value": 3.0   # maximum sample value
    }
}
```

## API Reference

### Job Class

#### Constructor
```python
Job(
    n_rows: int,
    model_info: Optional[dict] = None,
    dataset: Optional[dict] = None,
    save_filepath: Optional[str] = None,
    functions: Optional[list[dict]] = None
)
```

**Parameters:**
- `n_rows`: Number of synthetic rows to generate
- `model_info`: Model configuration dictionary
- `dataset`: Dataset configuration dictionary
- `save_filepath`: Path to save trained models
- `functions`: List of function configurations

#### Methods

##### train()
```python
train() -> tuple[list[dict], dict, UnspecializedModel, list[dict]]
```
Trains a model and generates synthetic data.

**Returns:**
- `results`: Generated synthetic data
- `metrics`: Quality evaluation metrics
- `model`: Trained model instance
- `schema`: Data schema information

##### infer()
```python
infer() -> tuple[list[dict], dict]
```
Generates data using a pre-trained model.

**Returns:**
- `results`: Generated synthetic data
- `metrics`: Quality metrics (if real data available)

##### generate_from_functions()
```python
generate_from_functions(dataset: Optional[Dataset] = None) -> list[dict]
```
Generates data using mathematical functions.

**Parameters:**
- `dataset`: Optional existing dataset to modify

**Returns:**
- `results`: Generated synthetic data

### Utility Functions

#### get_hyperparameters()
```python
get_hyperparameters() -> dict
```
Retrieves hyperparameters from environment variables.

**Returns:**
- Dictionary of hyperparameter settings

### Data Classes

#### Dataset
Abstract base class for all data types.

**Methods:**
- `from_json()`: Load data from JSON format
- `from_skeleton()`: Create from schema
- `preprocess()`: Apply preprocessing
- `postprocess()`: Apply postprocessing
- `to_json()`: Convert to JSON
- `to_skeleton()`: Extract schema

#### UnspecializedModel
Abstract base class for all models.

**Methods:**
- `train()`: Train the model
- `infer()`: Generate synthetic data
- `save()`: Save model to disk
- `load()`: Load model from disk
- `set_hyperparameters()`: Set model hyperparameters

## Examples and Use Cases

### Use Case 1: Customer Data Generation

Generate synthetic customer data for testing while preserving privacy.

```python
from sdg_core_lib import Job
import json

# Load real customer data configuration
with open("customers_config.json", "r") as f:
    dataset_config = json.load(f)

model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TabularVAE.TabularVAE",
    "model_name": "customer_synthetic_generator"
}

# Generate synthetic customers
job = Job(
    n_rows=1000,  # Generate 1000 synthetic customers
    model_info=model_config,
    dataset=dataset_config,
    save_filepath="./customer_models"
)

synthetic_customers, metrics, model, schema = job.train()

# Evaluate quality
print(f"Privacy Score: {metrics.get('privacy_score', 'N/A')}")
print(f"Statistical Similarity: {metrics.get('statistical_similarity', 'N/A')}")

# Save synthetic data
with open("synthetic_customers.json", "w") as f:
    json.dump(synthetic_customers, f)
```

### Use Case 2: Financial Time Series

Generate synthetic financial time series for backtesting trading strategies.

```python
from sdg_core_lib import Job
import json

# Financial time series data with proper structure
time_series_config = {
    "dataset_type": "time_series",
    "data": [
        {
            "column_name": "experiment_id",
            "column_type": "group_index",
            "column_data": [1, 1, 2, 2, 3, 4, 5],  # 5 experiments
            "column_datatype": "int"
        },
        {
            "column_name": "time",
            "column_type": "primary_key",
            "column_data": [0, 1, 2, 0, 1, 2],  # 3 time steps per experiment
            "column_datatype": "int"
        },
        {
            "column_name": "price",
            "column_type": "continuous",
            "column_data": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            "column_datatype": "float64"
        },
        {
            "column_name": "volume",
            "column_type": "continuous",
            "column_data": [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000],
            "column_datatype": "int64"
        }
    ]
}

# Configure time series generation
model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TimeSeriesVAE.TimeSeriesVAE",
    "model_name": "financial_ts_generator"
}

# Generate synthetic time series
job = Job(
    n_rows=500,  # 500 trading days
    model_info=model_config,
    dataset=time_series_config,
    save_filepath="./financial_models"
)

synthetic_ts, metrics, model, schema = job.train()

# Analyze results
print(f"Generated {len(synthetic_ts)} trading days")
print(f"First few entries: {synthetic_ts[:3]}")
```

### Use Case 3: Healthcare Data

Generate synthetic patient data for research while maintaining HIPAA compliance.

```python
from sdg_core_lib import Job

# Sample patient data (ensure no real PHI)
patient_data = [
    {
        "age": 45,
        "gender": "F",
        "blood_pressure": 120,
        "heart_rate": 72,
        "cholesterol": 190,
        "diabetes": 0,
        "smoker": 0
    },
    # ... more patient records
]

# Configure generation with privacy focus
dataset_config = {
    "dataset_type": "table",
    "data": patient_data
}

model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TabularVAE.TabularVAE",
    "model_name": "patient_data_generator"
}

# Generate synthetic patient data
job = Job(
    n_rows=1000,
    model_info=model_config,
    dataset=dataset_config,
    save_filepath="./healthcare_models"
)

synthetic_patients, metrics, model, schema = job.train()

# Validate privacy preservation
print(f"Privacy metrics: {metrics}")
print(f"Generated {len(synthetic_patients)} synthetic patient records")
```

### Use Case 4: Function-Based Data Generation

Generate controlled datasets for specific testing scenarios.

```python
from sdg_core_lib import Job

# Define mathematical relationships
functions = [
    {
        "feature": "experience_years",
        "function_name": "LinearFunction",
        "parameters": {
            "m": 1.0,
            "q": 0.0,
            "min_value": 0.0,
            "max_value": 20.0
        }
    },
    {
        "feature": "salary",
        "function_name": "QuadraticFunction",
        "parameters": {
            "a": 500.0,
            "b": 2000.0,
            "c": 30000.0,
            "min_value": 0.0,
            "max_value": 20.0
        }
    },
    {
        "feature": "performance_score",
        "function_name": "SinusoidalFunction",
        "parameters": {
            "amplitude": 20.0,
            "frequency": 0.5,
            "phase": 0.0,
            "min_value": 0.0,
            "max_value": 20.0
        }
    }
]

# Generate controlled dataset
job = Job(n_rows=100, functions=functions)

synthetic_data = job.generate_from_functions()

# Analyze relationships
print("Generated dataset:")
print(f"First few rows: {synthetic_data[:5]}")
```

### Use Case 5: Model Comparison

Compare different models to find the best for your data.

```python
from sdg_core_lib import Job
import json

# Load sample data configuration
with open('sample_data_config.json', 'r') as f:
    dataset_config = json.load(f)

# Test different models
models = [
    {
        "name": "TabularVAE",
        "config": {
            "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TabularVAE.TabularVAE",
            "model_name": "vae_model"
        }
    },
    {
        "name": "CTGAN",
        "config": {
            "algorithm_name": "sdg_core_lib.data_generator.models.GANs.implementation.CTGAN.CTGAN",
            "model_name": "ctgan_model"
        }
    }
]

results_comparison = {}

for model in models:
    job = Job(
        n_rows=200,
        model_info=model["config"],
        dataset=dataset_config,
        save_filepath=f"./{model['name'].lower()}_models"
    )
    
    synthetic_data, metrics, trained_model, schema = job.train()
    
    results_comparison[model["name"]] = {
        "statistical_similarity": metrics.get("statistical_similarity", 0),
        "correlation_preservation": metrics.get("correlation_preservation", 0),
        "privacy_score": metrics.get("privacy_score", 0)
    }

# Compare results
print("Model Comparison:")
for model_name, metrics in results_comparison.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")

# Find best model
best_model = max(results_comparison.keys(), 
                key=lambda x: results_comparison[x].get("statistical_similarity", 0))
print(f"\nBest model: {best_model}")
```

## Best Practices

### Data Preparation

#### 1. Data Quality
- Remove duplicates and outliers
- Handle missing values consistently
- Ensure consistent data types
- Validate data integrity

#### 2. Feature Engineering
- Normalize numeric features
- Encode categorical variables appropriately
- Create meaningful derived features
- Reduce high cardinality categories

#### 3. Data Size
- Minimum 1000 rows for model training
- More data generally improves quality
- Consider data augmentation for small datasets
- Balance class distribution

### Model Selection

#### 1. Start Simple
- Begin with TabularVAE for stability
- Progress to CTGAN for complexity
- Compare multiple models
- Use cross-validation when possible

#### 2. Hyperparameter Tuning
- Start with default hyperparameters
- Adjust learning rate if training fails
- Monitor training convergence
- Use early stopping to prevent overfitting

#### 3. Model Evaluation
- Use multiple quality metrics
- Consider downstream task performance
- Validate privacy preservation
- Perform visual inspection

### Performance Optimization

#### 1. Hardware Optimization
- Use GPU when available
- Enable mixed precision training
- Optimize batch size for memory
- Use parallel processing

#### 2. Software Optimization
- Use appropriate data types
- Optimize memory usage
- Cache preprocessed data
- Use efficient data structures

#### 3. Pipeline Optimization
- Stream large datasets
- Use incremental processing
- Optimize I/O operations
- Monitor resource usage

### Privacy Considerations

#### 1. Data Anonymization
- Remove direct identifiers
- Apply differential privacy
- Use aggregation techniques
- Validate privacy metrics

#### 2. Risk Assessment
- Evaluate re-identification risk
- Consider attack scenarios
- Implement access controls
- Document privacy measures

#### 3. Compliance
- Follow GDPR guidelines
- Meet industry regulations
- Document data usage
- Implement audit trails

## Troubleshooting

### Common Issues

#### 1. Installation Problems
**Problem**: Import errors or missing dependencies
**Solution**: 
```bash
# Reinstall with specific versions
pip uninstall sdg-core-lib
pip install sdg-core-lib==0.1.8

# Check Python version
python --version  # Should be 3.12+
```

#### 2. Memory Errors
**Problem**: Out of memory during training
**Solution**:
```python
# Reduce batch size
import os
os.environ["BATCH_SIZE"] = "16"

# Enable memory growth for GPU
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

#### 3. Poor Quality Results
**Problem**: Generated data doesn't match real data
**Solution**:
- Increase training epochs
- Try different model architectures
- Improve data preprocessing
- Use larger training dataset

#### 4. Training Instability
**Problem**: Model training fails or produces errors
**Solution**:
```python
# Adjust learning rate
os.environ["LEARNING_RATE"] = "0.0001"

# Use gradient clipping
os.environ["GRADIENT_CLIP_NORM"] = "1.0"

# Try different optimizer
model_config["optimizer"] = "adam"
```

#### 5. Slow Performance
**Problem**: Training takes too long
**Solution**:
- Enable GPU acceleration
- Increase batch size
- Use data parallelism
- Optimize data loading

### Debugging Techniques

#### 1. Logging
```python
import logging
logging.basicConfig(level=logging.INFO)

# Enable detailed logging
os.environ["GENESIS_LOG_LEVEL"] = "DEBUG"
```

#### 2. Data Validation
```python
# Validate input data
def validate_data(data):
    assert isinstance(data, list), "Data must be a list"
    assert len(data) > 0, "Data cannot be empty"
    assert all(isinstance(row, dict) for row in data), "Rows must be dictionaries"
    return True

validate_data(dataset_config["data"])
```

#### 3. Model Inspection
```python
# Inspect model architecture
model = job._model_factory()
print(f"Model input shape: {model.input_shape}")
print(f"Model parameters: {model.count_params()}")
```

#### 4. Quality Metrics
```python
# Detailed quality analysis
def analyze_quality(real_data, synthetic_data, metrics):
    print("Quality Analysis:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    # Visual comparison
    # Add visualization code here
```

### Performance Monitoring

#### 1. Resource Usage
```python
import psutil
import time

def monitor_resources():
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "rss": memory_info.rss,  # Physical memory
        "vms": memory_info.vms,  # Virtual memory
        "percent": process.memory_percent()
    }

def cleanup_memory():
    import gc
    gc.collect()
    # Clear TensorFlow session if needed
    tf.keras.backend.clear_session()
```

#### 2. Training Progress
```python
# Monitor training progress
os.environ["VERBOSE"] = "1"
os.environ["SHOW_PROGRESS"] = "true"
```

#### 3. Quality Tracking
```python
# Track quality over time
quality_history = []
for epoch in range(100):
    # Train for one epoch
    # Evaluate quality
    quality_history.append(current_quality)
    
    # Plot quality progression
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Quality = {current_quality:.3f}")
```

## Advanced Topics

### Custom Model Development

#### Creating Custom Models
```python
from sdg_core_lib.data_generator.models.UnspecializedModel import UnspecializedModel

class CustomModel(UnspecializedModel):
    def _build(self, input_shape):
        # Implement model architecture
        pass
    
    def train(self, data):
        # Implement training logic
        pass
    
    def infer(self, n_rows):
        # Implement inference logic
        pass
    
    def save(self, folder_path):
        # Implement model saving
        pass
    
    def load(self, model_filepath):
        # Implement model loading
        pass
```

### Custom Function Development

#### Creating Custom Functions
```python
from sdg_core_lib.post_process.functions.UnspecializedFunction import UnspecializedFunction
from sdg_core_lib.post_process.functions.Parameter import Parameter

class CustomFunction(UnspecializedFunction):
    parameters = [
        Parameter("param1", "default_value", "float"),
        Parameter("param2", "default_value", "int")
    ]
    description = "Custom function description"
    priority = Priority.MEDIUM
    is_generative = True
    
    def apply(self, n_rows, data):
        # Implement function logic
        return processed_data
```

### Integration with ML Pipelines

#### Scikit-learn Integration
```python
from sklearn.base import BaseEstimator
from sdg_core_lib import Job

class SyntheticDataGenerator(BaseEstimator):
    def __init__(self, n_rows=1000, model_type="VAE"):
        self.n_rows = n_rows
        self.model_type = model_type
    
    def fit(self, X, y=None):
        # Train synthetic data generator
        pass
    
    def transform(self, X):
        # Generate synthetic data
        pass
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
```

#### TensorFlow Integration
```python
import tensorflow as tf
from sdg_core_lib import Job

# Create TensorFlow dataset
tf_dataset = tf.data.Dataset.from_tensor_slices(real_data)

# Integrate with GENESIS
class TensorFlowSyntheticGenerator:
    def __init__(self, job_config):
        self.job = Job(**job_config)
    
    def generate_dataset(self, n_samples):
        synthetic_data, _, _, _ = self.job.train()
        return tf.data.Dataset.from_tensor_slices(synthetic_data)
```

This comprehensive user documentation provides everything you need to effectively use GENESIS Core Lib for synthetic data generation across various use cases and scenarios.
