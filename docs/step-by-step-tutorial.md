# Step-by-Step Tutorial

## Overview

This comprehensive tutorial will guide you through using GENESIS Core Lib from basic concepts to advanced applications. You'll learn by doing, with practical examples and real-world use cases.

## Prerequisites

- Python 3.12 or higher
- Basic understanding of Python and data concepts
- GENESIS Core Lib installed (see [Installation Guide](standalone-installation.md))

## Tutorial Structure

1. [Getting Started](#getting-started)
2. [Basic Data Generation](#basic-data-generation)
3. [Working with Real Data](#working-with-real-data)
4. [Advanced Model Configuration](#advanced-model-configuration)
5. [Function-Based Generation](#function-based-generation)
6. [Quality Evaluation](#quality-evaluation)
7. [Real-World Use Case](#real-world-use-case)
8. [Troubleshooting and Optimization](#troubleshooting-and-optimization)

## Getting Started

### Step 1: Verify Installation

First, let's ensure GENESIS Core Lib is properly installed:

```python
# Test basic import
from sdg_core_lib import Job
print("✅ GENESIS Core Lib imported successfully!")

# Check version
import pkg_resources
version = pkg_resources.get_distribution("sdg-core-lib").version
print(f"Version: {version}")
```

**Expected Output:**
```
✅ GENESIS Core Lib imported successfully!
Version: 0.1.8
```

### Step 2: Understand the Core Concepts

GENESIS Core Lib works with these main components:

1. **Job**: The main orchestrator
2. **Dataset**: Your input data
3. **Model**: The ML model for generation
4. **Functions**: Mathematical functions for data creation

Let's create a simple example to understand the flow:

```python
# This is just a conceptual overview - we'll implement it step by step
"""
1. Define your data (Dataset)
2. Choose a model (Model)
3. Create a job (Job)
4. Train and generate (train/infer)
5. Evaluate results (Metrics)
"""
```

## Basic Data Generation

### Step 3: Your First Synthetic Dataset

Let's start with the simplest possible example - generating synthetic tabular data:

```python
from sdg_core_lib import Job

# Step 3a: Define your input data
sample_data = [
    {"age": 25, "income": 50000, "city": "New York"},
    {"age": 30, "income": 75000, "city": "San Francisco"},
    {"age": 35, "income": 90000, "city": "Chicago"},
    {"age": 28, "income": 60000, "city": "Boston"},
    {"age": 32, "income": 80000, "city": "Seattle"}
]

# Step 3b: Configure the dataset
dataset_config = {
    "dataset_type": "table",
    "data": sample_data
}

# Step 3c: Choose a model
model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.GANs.TabularGAN",
    "model_name": "my_first_model"
}

# Step 3d: Create the job
job = Job(
    n_rows=20,  # Generate 20 synthetic rows
    model_info=model_config,
    dataset=dataset_config,
    save_filepath="./tutorial_models"
)

print("Job created successfully!")
print(f"Will generate {job._Job__n_rows} synthetic rows")
```

### Step 4: Train and Generate Data

Now let's actually train the model and generate synthetic data:

```python
# Step 4a: Train the model and generate data
print("🚀 Starting training...")
results, metrics, model, schema = job.train()
print("✅ Training completed!")

# Step 4b: Examine the results
print(f"\n📊 Generated {len(results)} synthetic rows:")
print("First 5 rows:")
for i, row in enumerate(results[:5]):
    print(f"  Row {i+1}: {row}")

# Step 4c: Check quality metrics
print(f"\n📈 Quality Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.3f}" if isinstance(value, (int, float)) else f"  {metric}: {value}")
```

**Expected Output (example):**
```
🚀 Starting training...
✅ Training completed!

📊 Generated 20 synthetic rows:
First 5 rows:
  Row 1: {'age': 27.3, 'income': 62000, 'city': 'New York'}
  Row 2: {'age': 31.8, 'income': 78000, 'city': 'San Francisco'}
  Row 3: {'age': 29.1, 'income': 65000, 'city': 'Chicago'}
  Row 4: {'age': 26.5, 'income': 58000, 'city': 'Boston'}
  Row 5: {'age': 33.2, 'income': 82000, 'city': 'Seattle'}

📈 Quality Metrics:
  statistical_similarity: 0.856
  correlation_preservation: 0.923
  privacy_score: 0.891
```

### Step 5: Analyze the Results

Let's do a more detailed analysis of what we generated:

```python
import pandas as pd
import numpy as np

# Convert to pandas DataFrame for easier analysis
original_df = pd.DataFrame(sample_data)
synthetic_df = pd.DataFrame(results)

print("🔍 Data Analysis:")
print(f"\nOriginal data shape: {original_df.shape}")
print(f"Synthetic data shape: {synthetic_df.shape}")

print("\n📋 Original Data Summary:")
print(original_df.describe())

print("\n📋 Synthetic Data Summary:")
print(synthetic_df.describe())

print("\n🏙️ City Distribution:")
print("Original:", original_df['city'].value_counts().to_dict())
print("Synthetic:", synthetic_df['city'].value_counts().to_dict())
```

## Working with Real Data

### Step 6: Load and Prepare Real Data

Let's work with a more realistic dataset. For this tutorial, we'll create a sample that mimics real customer data:

```python
import pandas as pd
import numpy as np

# Create a more realistic dataset
np.random.seed(42)  # For reproducible results

n_customers = 200
real_data = []

for i in range(n_customers):
    customer = {
        "customer_id": i + 1,
        "age": np.random.randint(18, 80),
        "income": np.random.normal(60000, 25000),
        "credit_score": np.random.randint(300, 850),
        "years_with_bank": np.random.randint(0, 20),
        "has_credit_card": np.random.choice([0, 1], p=[0.3, 0.7]),
        "account_balance": np.random.exponential(5000),
        "city": np.random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"])
    }
    real_data.append(customer)

# Convert to DataFrame and clean
real_df = pd.DataFrame(real_data)
real_df['income'] = np.maximum(real_df['income'], 20000)  # Minimum income
real_df['account_balance'] = np.round(real_df['account_balance'], 2)

print("📊 Real Customer Dataset Created:")
print(f"Shape: {real_df.shape}")
print("\nSample data:")
print(real_df.head())
print("\nData types:")
print(real_df.dtypes)
```

### Step 7: Preprocess and Configure

Now let's prepare this data for synthetic generation:

```python
# Convert to required format
data_payload = real_df.to_dict("records")

# Configure dataset
dataset_config = {
    "dataset_type": "table",
    "data": data_payload
}

# Choose model - let's try VAE this time for variety
model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.TabularVAE",
    "model_name": "customer_vae_model"
}

# Create job with more rows for better training
job = Job(
    n_rows=500,  # Generate 500 synthetic customers
    model_info=model_config,
    dataset=dataset_config,
    save_filepath="./customer_models"
)

print("✅ Customer data job configured")
print(f"Training on {len(data_payload)} real customers")
print(f"Will generate {job._Job__n_rows} synthetic customers")
```

### Step 8: Train and Evaluate

```python
# Train the model
print("🎯 Training customer data model...")
synthetic_customers, metrics, model, schema = job.train()
print("✅ Training completed!")

# Convert to DataFrame for analysis
synthetic_customer_df = pd.DataFrame(synthetic_customers)

print(f"\n📊 Generated {len(synthetic_customer_df)} synthetic customers")
print("\n📋 Synthetic Customer Data Summary:")
print(synthetic_customer_df.describe())

print(f"\n📈 Quality Metrics:")
for metric, value in metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {metric}: {value:.3f}")
    else:
        print(f"  {metric}: {value}")
```

### Step 9: Compare Distributions

Let's visually compare the real and synthetic data:

```python
import matplotlib.pyplot as plt

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Real vs Synthetic Data Comparison', fontsize=16)

# Age comparison
axes[0, 0].hist(real_df['age'], bins=20, alpha=0.7, label='Real', color='blue')
axes[0, 0].hist(synthetic_customer_df['age'], bins=20, alpha=0.7, label='Synthetic', color='red')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].legend()

# Income comparison
axes[0, 1].hist(real_df['income'], bins=20, alpha=0.7, label='Real', color='blue')
axes[0, 1].hist(synthetic_customer_df['income'], bins=20, alpha=0.7, label='Synthetic', color='red')
axes[0, 1].set_title('Income Distribution')
axes[0, 1].legend()

# Credit score comparison
axes[0, 2].hist(real_df['credit_score'], bins=20, alpha=0.7, label='Real', color='blue')
axes[0, 2].hist(synthetic_customer_df['credit_score'], bins=20, alpha=0.7, label='Synthetic', color='red')
axes[0, 2].set_title('Credit Score Distribution')
axes[0, 2].legend()

# Account balance comparison
axes[1, 0].hist(real_df['account_balance'], bins=20, alpha=0.7, label='Real', color='blue')
axes[1, 0].hist(synthetic_customer_df['account_balance'], bins=20, alpha=0.7, label='Synthetic', color='red')
axes[1, 0].set_title('Account Balance Distribution')
axes[1, 0].legend()

# City comparison
real_city_counts = real_df['city'].value_counts()
synthetic_city_counts = synthetic_customer_df['city'].value_counts()
axes[1, 1].bar(real_city_counts.index, real_city_counts.values, alpha=0.7, label='Real', color='blue')
axes[1, 1].bar(synthetic_city_counts.index, synthetic_city_counts.values, alpha=0.7, label='Synthetic', color='red')
axes[1, 1].set_title('City Distribution')
axes[1, 1].legend()

# Credit card comparison
real_cc_counts = real_df['has_credit_card'].value_counts()
synthetic_cc_counts = synthetic_customer_df['has_credit_card'].value_counts()
axes[1, 2].bar(['No', 'Yes'], [real_cc_counts.get(0, 0), real_cc_counts.get(1, 0)], alpha=0.7, label='Real', color='blue')
axes[1, 2].bar(['No', 'Yes'], [synthetic_cc_counts.get(0, 0), synthetic_cc_counts.get(1, 0)], alpha=0.7, label='Synthetic', color='red')
axes[1, 2].set_title('Credit Card Ownership')
axes[1, 2].legend()

plt.tight_layout()
plt.show()

print("📊 Comparison plots generated!")
```

## Advanced Model Configuration

### Step 10: Custom Hyperparameters

Let's explore how to customize model training:

```python
import os

# Set custom hyperparameters
os.environ["EPOCHS"] = "200"  # Increase training epochs
os.environ["LEARNING_RATE"] = "0.001"  # Adjust learning rate
os.environ["BATCH_SIZE"] = "64"  # Increase batch size

print("🔧 Custom hyperparameters set:")
print(f"  EPOCHS: {os.environ.get('EPOCHS')}")
print(f"  LEARNING_RATE: {os.environ.get('LEARNING_RATE')}")
print(f"  BATCH_SIZE: {os.environ.get('BATCH_SIZE')}")

# Create job with custom settings
job_custom = Job(
    n_rows=300,
    model_info={
        "algorithm_name": "sdg_core_lib.data_generator.models.GANs.TabularGAN",
        "model_name": "custom_gan_model"
    },
    dataset=dataset_config,
    save_filepath="./custom_models"
)

print("\n🚀 Training with custom hyperparameters...")
synthetic_custom, custom_metrics, custom_model, custom_schema = job_custom.train()
print("✅ Custom training completed!")

print(f"\n📈 Custom Model Quality Metrics:")
for metric, value in custom_metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {metric}: {value:.3f}")
    else:
        print(f"  {metric}: {value}")
```

### Step 11: Model Comparison

Let's compare different models to see which performs best:

```python
# Test multiple models
models_to_test = [
    {
        "name": "TabularGAN",
        "config": {
            "algorithm_name": "sdg_core_lib.data_generator.models.GANs.TabularGAN",
            "model_name": "comparison_gan"
        }
    },
    {
        "name": "TabularVAE",
        "config": {
            "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.TabularVAE",
            "model_name": "comparison_vae"
        }
    }
]

comparison_results = {}

for model_info in models_to_test:
    print(f"\n🔄 Testing {model_info['name']}...")
    
    job = Job(
        n_rows=200,
        model_info=model_info["config"],
        dataset=dataset_config,
        save_filepath=f"./comparison_{model_info['name'].lower()}_models"
    )
    
    synthetic_data, metrics, trained_model, schema = job.train()
    
    comparison_results[model_info["name"]] = metrics
    print(f"✅ {model_info['name']} completed")

# Create comparison table
print("\n📊 Model Comparison:")
print("=" * 50)
for model_name, metrics in comparison_results.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")

# Find best model based on statistical similarity
best_model = max(comparison_results.keys(), 
                key=lambda x: comparison_results[x].get('statistical_similarity', 0))
print(f"\n🏆 Best model: {best_model}")
```

## Function-Based Generation

### Step 12: Mathematical Function Generation

Now let's explore function-based data generation, which gives you precise control over the data:

```python
from sdg_core_lib import Job

# Define mathematical functions for data generation
functions = [
    {
        "feature": "experience_years",
        "function_name": "LinearFunction",
        "parameters": {
            "m": 1.0,      # slope
            "q": 0.0,      # y-intercept
            "min_value": 0.0,
            "max_value": 20.0
        }
    },
    {
        "feature": "salary",
        "function_name": "QuadraticFunction",
        "parameters": {
            "a": 800.0,    # quadratic coefficient
            "b": 3000.0,   # linear coefficient
            "c": 35000.0,  # constant term
            "min_value": 0.0,
            "max_value": 20.0
        }
    },
    {
        "feature": "performance_score",
        "function_name": "SinusoidalFunction",
        "parameters": {
            "amplitude": 15.0,
            "frequency": 0.3,
            "phase": 0.0,
            "min_value": 0.0,
            "max_value": 20.0
        }
    },
    {
        "feature": "satisfaction_level",
        "function_name": "NormalDistributionSample",
        "parameters": {
            "mean": 7.0,
            "std_dev": 1.5,
            "min_value": 1.0,
            "max_value": 10.0
        }
    }
]

# Create job with function-based generation
job_functions = Job(n_rows=100, functions=functions)

print("🔧 Function-based generation configured:")
print("Functions:")
for func in functions:
    print(f"  - {func['feature']}: {func['function_name']}")

# Generate data
print("\n🚀 Generating data from functions...")
synthetic_function_data = job_functions.generate_from_functions()
print("✅ Function-based generation completed!")

# Analyze the generated data
function_df = pd.DataFrame(synthetic_function_data)
print(f"\n📊 Generated {len(function_df)} rows")
print("\n📋 Data Summary:")
print(function_df.describe())

print("\n🔗 Correlations:")
print(function_df.corr())
```

### Step 13: Visualize Function-Generated Data

```python
# Create plots to visualize the mathematical relationships
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Function-Generated Data Relationships', fontsize=16)

# Experience vs Salary (should be quadratic)
axes[0, 0].scatter(function_df['experience_years'], function_df['salary'], alpha=0.6)
axes[0, 0].set_title('Experience vs Salary')
axes[0, 0].set_xlabel('Experience (years)')
axes[0, 0].set_ylabel('Salary ($)')

# Experience vs Performance (should be sinusoidal)
axes[0, 1].scatter(function_df['experience_years'], function_df['performance_score'], alpha=0.6)
axes[0, 1].set_title('Experience vs Performance Score')
axes[0, 1].set_xlabel('Experience (years)')
axes[0, 1].set_ylabel('Performance Score')

# Performance vs Satisfaction
axes[1, 0].scatter(function_df['performance_score'], function_df['satisfaction_level'], alpha=0.6)
axes[1, 0].set_title('Performance vs Satisfaction')
axes[1, 0].set_xlabel('Performance Score')
axes[1, 0].set_ylabel('Satisfaction Level')

# Distribution of Satisfaction
axes[1, 1].hist(function_df['satisfaction_level'], bins=20, alpha=0.7, color='green')
axes[1, 1].set_title('Satisfaction Level Distribution')
axes[1, 1].set_xlabel('Satisfaction Level')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

print("📊 Function relationship plots generated!")
```

## Quality Evaluation

### Step 14: Comprehensive Quality Analysis

Let's dive deeper into evaluating synthetic data quality:

```python
def comprehensive_quality_analysis(real_data, synthetic_data):
    """Perform detailed quality analysis"""
    
    real_df = pd.DataFrame(real_data)
    synthetic_df = pd.DataFrame(synthetic_data)
    
    analysis = {}
    
    # 1. Statistical Similarity
    for column in real_df.select_dtypes(include=[np.number]).columns:
        if column in synthetic_df.columns:
            # Kolmogorov-Smirnov test
            from scipy.stats import ks_2samp
            ks_stat, p_value = ks_2samp(real_df[column], synthetic_df[column])
            
            # Mean and variance comparison
            mean_diff = abs(real_df[column].mean() - synthetic_df[column].mean())
            var_diff = abs(real_df[column].var() - synthetic_df[column].var())
            
            analysis[column] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'mean_difference': mean_diff,
                'variance_difference': var_diff
            }
    
    # 2. Correlation Preservation
    real_corr = real_df.select_dtypes(include=[np.number]).corr()
    synthetic_corr = synthetic_df.select_dtypes(include=[np.number]).corr()
    
    if not real_corr.empty and not synthetic_corr.empty:
        corr_diff = np.abs(real_corr - synthetic_corr)
        mean_corr_diff = corr_diff.mean().mean()
        analysis['correlation_preservation'] = {
            'mean_difference': mean_corr_diff,
            'correlation_matrix_diff': corr_diff
        }
    
    # 3. Categorical Distribution
    for column in real_df.select_dtypes(include=['object']).columns:
        if column in synthetic_df.columns:
            real_counts = real_df[column].value_counts(normalize=True)
            synthetic_counts = synthetic_df[column].value_counts(normalize=True)
            
            # Calculate distribution difference
            all_categories = set(real_counts.index) | set(synthetic_counts.index)
            real_aligned = real_counts.reindex(all_categories, fill_value=0)
            synthetic_aligned = synthetic_counts.reindex(all_categories, fill_value=0)
            
            dist_diff = np.abs(real_aligned - synthetic_aligned).sum() / 2
            analysis[f'{column}_distribution'] = {
                'distribution_difference': dist_diff,
                'real_distribution': real_aligned.to_dict(),
                'synthetic_distribution': synthetic_aligned.to_dict()
            }
    
    return analysis

# Perform comprehensive analysis
print("🔍 Performing comprehensive quality analysis...")
quality_analysis = comprehensive_quality_analysis(real_data, synthetic_customers)

print("✅ Analysis completed!")
print("\n📊 Detailed Quality Results:")

# Display results
for metric_name, results in quality_analysis.items():
    print(f"\n{metric_name}:")
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, dict):
                print(f"  {key}: {len(value)} items")
            else:
                print(f"  {key}: {type(value).__name__}")
```

### Step 15: Privacy Assessment

Let's evaluate how well the synthetic data preserves privacy:

```python
def privacy_assessment(real_data, synthetic_data, sensitive_columns=None):
    """Assess privacy preservation"""
    
    if sensitive_columns is None:
        sensitive_columns = ['customer_id', 'age', 'income']  # Example sensitive columns
    
    real_df = pd.DataFrame(real_data)
    synthetic_df = pd.DataFrame(synthetic_data)
    
    privacy_metrics = {}
    
    for column in sensitive_columns:
        if column in real_df.columns and column in synthetic_df.columns:
            # Check for exact matches
            exact_matches = set(real_df[column]) & set(synthetic_df[column])
            match_percentage = len(exact_matches) / len(set(real_df[column])) * 100
            
            # Check distribution similarity (lower is better for privacy)
            from scipy.stats import wasserstein_distance
            real_values = real_df[column].dropna()
            synthetic_values = synthetic_df[column].dropna()
            
            if len(real_values) > 0 and len(synthetic_values) > 0:
                emd = wasserstein_distance(real_values, synthetic_values)
            else:
                emd = 0
            
            privacy_metrics[column] = {
                'exact_match_percentage': match_percentage,
                'earth_movers_distance': emd,
                'privacy_score': max(0, 100 - match_percentage)  # Simple privacy score
            }
    
    return privacy_metrics

# Perform privacy assessment
print("🔒 Performing privacy assessment...")
privacy_metrics = privacy_assessment(real_data, synthetic_customers)

print("✅ Privacy assessment completed!")
print("\n🛡️ Privacy Metrics:")

for column, metrics in privacy_metrics.items():
    print(f"\n{column}:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value}")
```

## Real-World Use Case

### Step 16: Complete Healthcare Data Example

Let's work through a realistic healthcare scenario:

```python
# Create synthetic healthcare dataset
np.random.seed(123)

n_patients = 300
healthcare_data = []

for i in range(n_patients):
    patient = {
        "patient_id": f"P{i+1:04d}",
        "age": np.random.randint(18, 85),
        "gender": np.random.choice(["M", "F"], p=[0.48, 0.52]),
        "bmi": np.random.normal(28.5, 5.2),
        "blood_pressure_systolic": np.random.normal(120, 15),
        "blood_pressure_diastolic": np.random.normal(80, 10),
        "cholesterol": np.random.normal(200, 35),
        "heart_rate": np.random.normal(72, 8),
        "has_diabetes": np.random.choice([0, 1], p=[0.85, 0.15]),
        "smoker": np.random.choice([0, 1], p=[0.75, 0.25]),
        "exercise_hours_per_week": np.random.exponential(3),
        "last_visit_days_ago": np.random.randint(1, 365)
    }
    
    # Ensure realistic values
    patient["bmi"] = np.clip(patient["bmi"], 15, 50)
    patient["blood_pressure_systolic"] = np.clip(patient["blood_pressure_systolic"], 80, 200)
    patient["blood_pressure_diastolic"] = np.clip(patient["blood_pressure_diastolic"], 50, 120)
    patient["cholesterol"] = np.clip(patient["cholesterol"], 100, 400)
    patient["heart_rate"] = np.clip(patient["heart_rate"], 40, 120)
    patient["exercise_hours_per_week"] = np.clip(patient["exercise_hours_per_week"], 0, 20)
    
    healthcare_data.append(patient)

# Create DataFrame
healthcare_df = pd.DataFrame(healthcare_data)

print("🏥 Healthcare Dataset Created:")
print(f"Shape: {healthcare_df.shape}")
print("\nSample data:")
print(healthcare_df.head())

print("\n📊 Patient Demographics:")
print(f"Average age: {healthcare_df['age'].mean():.1f} years")
print(f"Gender distribution: {healthcare_df['gender'].value_counts().to_dict()}")
print(f"Diabetes prevalence: {healthcare_df['has_diabetes'].mean()*100:.1f}%")
print(f"Smoking rate: {healthcare_df['smoker'].mean()*100:.1f}%")
```

### Step 17: Generate Synthetic Healthcare Data

```python
# Configure healthcare data generation
healthcare_config = {
    "dataset_type": "table",
    "data": healthcare_df.to_dict("records")
}

# Use VAE for healthcare data (often more stable)
healthcare_model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.TabularVAE",
    "model_name": "healthcare_vae_model"
}

# Create and run job
healthcare_job = Job(
    n_rows=500,  # Generate more patients
    model_info=healthcare_model_config,
    dataset=healthcare_config,
    save_filepath="./healthcare_models"
)

print("🏥 Training healthcare data model...")
synthetic_patients, healthcare_metrics, healthcare_model, healthcare_schema = healthcare_job.train()
print("✅ Healthcare model training completed!")

# Analyze synthetic healthcare data
synthetic_healthcare_df = pd.DataFrame(synthetic_patients)

print(f"\n👥 Generated {len(synthetic_healthcare_df)} synthetic patients")
print("\n📊 Synthetic Patient Demographics:")
print(f"Average age: {synthetic_healthcare_df['age'].mean():.1f} years")
print(f"Gender distribution: {synthetic_healthcare_df['gender'].value_counts().to_dict()}")
print(f"Diabetes prevalence: {synthetic_healthcare_df['has_diabetes'].mean()*100:.1f}%")
print(f"Smoking rate: {synthetic_healthcare_df['smoker'].mean()*100:.1f}%")
```

### Step 18: Healthcare Data Validation

```python
# Validate healthcare data quality
def validate_healthcare_data(real_df, synthetic_df):
    """Perform healthcare-specific validation"""
    
    validation_results = {}
    
    # 1. Vital signs ranges
    vital_ranges = {
        'blood_pressure_systolic': (80, 200),
        'blood_pressure_diastolic': (50, 120),
        'heart_rate': (40, 120),
        'bmi': (15, 50),
        'cholesterol': (100, 400)
    }
    
    for vital, (min_val, max_val) in vital_ranges.items():
        real_outliers = ((real_df[vital] < min_val) | (real_df[vital] > max_val)).sum()
        synthetic_outliers = ((synthetic_df[vital] < min_val) | (synthetic_df[vital] > max_val)).sum()
        
        validation_results[f"{vital}_outliers"] = {
            "real": real_outliers,
            "synthetic": synthetic_outliers,
            "real_percentage": real_outliers / len(real_df) * 100,
            "synthetic_percentage": synthetic_outliers / len(synthetic_df) * 100
        }
    
    # 2. Disease prevalence
    diseases = ['has_diabetes', 'smoker']
    for disease in diseases:
        real_prev = real_df[disease].mean()
        synthetic_prev = synthetic_df[disease].mean()
        prev_diff = abs(real_prev - synthetic_prev)
        
        validation_results[f"{disease}_prevalence"] = {
            "real": real_prev * 100,
            "synthetic": synthetic_prev * 100,
            "difference": prev_diff * 100
        }
    
    # 3. Age-gender distribution
    real_age_by_gender = real_df.groupby('gender')['age'].describe()
    synthetic_age_by_gender = synthetic_df.groupby('gender')['age'].describe()
    
    validation_results["age_by_gender"] = {
        "real": real_age_by_gender.to_dict(),
        "synthetic": synthetic_age_by_gender.to_dict()
    }
    
    return validation_results

# Perform validation
print("🔍 Validating healthcare data...")
healthcare_validation = validate_healthcare_data(healthcare_df, synthetic_healthcare_df)

print("✅ Healthcare validation completed!")
print("\n🏥 Healthcare Validation Results:")

for metric, results in healthcare_validation.items():
    print(f"\n{metric}:")
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.2f}")
            elif isinstance(value, dict):
                print(f"  {key}: Available")
            else:
                print(f"  {key}: {value}")
```

## Troubleshooting and Optimization

### Step 19: Common Issues and Solutions

Let's address common problems you might encounter:

```python
def troubleshoot_generation_job(job_config):
    """Diagnose and suggest fixes for common issues"""
    
    issues = []
    suggestions = []
    
    # Check data size
    dataset = job_config.get('dataset', {})
    data = dataset.get('data', [])
    
    if len(data) < 100:
        issues.append("Small training dataset")
        suggestions.append("Consider using more training data or function-based generation")
    
    if len(data) > 10000:
        issues.append("Very large dataset")
        suggestions.append("Consider using a smaller sample or batch processing")
    
    # Check data quality
    if data:
        # Check for missing values
        missing_values = sum(1 for row in data if None in row.values())
        if missing_values > 0:
            issues.append(f"Missing values detected: {missing_values}")
            suggestions.append("Handle missing values before training")
        
        # Check categorical cardinality
        categorical_columns = {}
        for row in data:
            for key, value in row.items():
                if isinstance(value, str):
                    if key not in categorical_columns:
                        categorical_columns[key] = set()
                    categorical_columns[key].add(value)
        
        high_cardinality = [col for col, values in categorical_columns.items() if len(values) > 50]
        if high_cardinality:
            issues.append(f"High cardinality columns: {high_cardinality}")
            suggestions.append("Consider reducing categories or encoding differently")
    
    # Check model configuration
    model_info = job_config.get('model_info', {})
    if not model_info:
        issues.append("No model configuration")
        suggestions.append("Specify model_info with algorithm_name and model_name")
    
    return issues, suggestions

# Test troubleshooting
test_job_config = {
    'dataset': healthcare_config,
    'model_info': healthcare_model_config
}

issues, suggestions = troubleshoot_generation_job(test_job_config)

print("🔧 Troubleshooting Results:")
if issues:
    print("\n⚠️ Issues found:")
    for issue in issues:
        print(f"  - {issue}")
    
    print("\n💡 Suggestions:")
    for suggestion in suggestions:
        print(f"  - {suggestion}")
else:
    print("✅ No issues detected!")
```

### Step 20: Performance Optimization

Let's optimize for better performance:

```python
def optimize_job_performance():
    """Provide performance optimization tips"""
    
    optimizations = []
    
    # Memory optimizations
    optimizations.append({
        "category": "Memory",
        "tip": "Use smaller batch sizes for large datasets",
        "implementation": "os.environ['BATCH_SIZE'] = '16'"
    })
    
    optimizations.append({
        "category": "Memory",
        "tip": "Enable memory growth for GPU",
        "implementation": "tf.config.experimental.set_memory_growth(gpu, True)"
    })
    
    # Speed optimizations
    optimizations.append({
        "category": "Speed",
        "tip": "Use mixed precision training",
        "implementation": "tf.keras.mixed_precision.set_global_policy('mixed_float16')"
    })
    
    optimizations.append({
        "category": "Speed",
        "tip": "Increase batch size if memory allows",
        "implementation": "os.environ['BATCH_SIZE'] = '64'"
    })
    
    # Quality optimizations
    optimizations.append({
        "category": "Quality",
        "tip": "Increase training epochs for better quality",
        "implementation": "os.environ['EPOCHS'] = '500'"
    })
    
    optimizations.append({
        "category": "Quality",
        "tip": "Adjust learning rate for better convergence",
        "implementation": "os.environ['LEARNING_RATE'] = '0.0001'"
    })
    
    return optimizations

# Display optimizations
optimizations = optimize_job_performance()

print("⚡ Performance Optimization Guide:")
for opt in optimizations:
    print(f"\n{opt['category']}:")
    print(f"  Tip: {opt['tip']}")
    print(f"  Code: {opt['implementation']}")
```

### Step 21: Best Practices Summary

```python
def print_best_practices():
    """Summarize best practices for using GENESIS Core Lib"""
    
    practices = [
        {
            "area": "Data Preparation",
            "practices": [
                "Use at least 1000 rows for model training",
                "Handle missing values consistently",
                "Limit categorical cardinality (<50 categories)",
                "Normalize numeric features when possible"
            ]
        },
        {
            "area": "Model Selection",
            "practices": [
                "Start with VAE for stability",
                "Use GAN for complex patterns",
                "Compare multiple models",
                "Adjust hyperparameters based on data size"
            ]
        },
        {
            "area": "Quality Assurance",
            "practices": [
                "Always evaluate synthetic data quality",
                "Check statistical similarity",
                "Verify privacy preservation",
                "Perform visual inspection"
            ]
        },
        {
            "area": "Performance",
            "practices": [
                "Use GPU for large datasets",
                "Optimize batch size for memory",
                "Monitor training progress",
                "Cache trained models when possible"
            ]
        },
        {
            "area": "Privacy",
            "practices": [
                "Remove direct identifiers",
                "Apply differential privacy when needed",
                "Validate privacy metrics",
                "Follow data protection regulations"
            ]
        }
    ]
    
    print("📚 GENESIS Core Lib Best Practices:")
    print("=" * 50)
    
    for area_info in practices:
        print(f"\n🎯 {area_info['area']}:")
        for practice in area_info['practices']:
            print(f"  ✓ {practice}")

print_best_practices()
```

## Conclusion

### Step 22: Tutorial Summary

Congratulations! You've completed the comprehensive GENESIS Core Lib tutorial. Let's summarize what you've learned:

```python
def tutorial_summary():
    """Provide a summary of what was learned in this tutorial"""
    
    skills_learned = [
        "Basic synthetic data generation using Job class",
        "Working with real-world datasets",
        "Advanced model configuration and hyperparameter tuning",
        "Function-based data generation with mathematical functions",
        "Comprehensive quality evaluation and privacy assessment",
        "Real-world healthcare data use case",
        "Troubleshooting and performance optimization",
        "Best practices for synthetic data generation"
    ]
    
    next_steps = [
        "Explore the [User Documentation](user-documentation.md) for advanced features",
        "Check the [Developer Documentation](developer-documentation.md) for extending the library",
        "Review the [API Reference](#api-reference) for detailed method information",
        "Join the community forums for support and discussions",
        "Contribute to the project on GitHub"
    ]
    
    print("🎓 Tutorial Completion Summary:")
    print("=" * 40)
    
    print("\n🚀 Skills You've Learned:")
    for i, skill in enumerate(skills_learned, 1):
        print(f"  {i}. {skill}")
    
    print("\n📈 Next Steps:")
    for i, step in enumerate(next_steps, 1):
        print(f"  {i}. {step}")
    
    print("\n💡 Key Takeaways:")
    print("  • Synthetic data generation is powerful for privacy and testing")
    print("  • Quality evaluation is essential for trustworthy synthetic data")
    print("  • Different models work better for different data types")
    print("  • Function-based generation gives you precise control")
    print("  • Always consider privacy implications")

tutorial_summary()
```

### Step 23: Final Project

As a final exercise, create a synthetic data generation project for your own use case:

```python
def final_project_template():
    """Template for your final project"""
    
    project_template = """
# Final Project: Your Synthetic Data Use Case

## 1. Define Your Use Case
# What type of data do you want to generate?
# Why do you need synthetic data?
# What are your privacy requirements?

## 2. Prepare Your Data
# Load or create your real dataset
# Clean and preprocess the data
# Validate data quality

## 3. Configure Generation
# Choose appropriate model type
# Set hyperparameters
# Define quality metrics

## 4. Generate Synthetic Data
# Train the model
# Generate synthetic dataset
# Evaluate quality

## 5. Validate and Deploy
# Perform quality assessment
# Check privacy preservation
# Prepare for production use

## 6. Document and Share
# Document your process
# Share results with stakeholders
# Get feedback and iterate
"""
    
    print("🎯 Final Project Template:")
    print(project_template)

final_project_template()
```

## Additional Resources

### Documentation Links
- [Main README](../README.md)
- [User Documentation](user-documentation.md)
- [Developer Documentation](developer-documentation.md)
- [API Reference](#api-reference)

### Community Resources
- [GitHub Repository](https://github.com/emiliocimino/generator_core_lib)
- [Issue Tracker](https://github.com/emiliocimino/generator_core_lib/issues)
- [Discussions](https://github.com/emiliocimino/generator_core_lib/discussions)

### Examples and Templates
- Check the `examples/` directory for more use cases
- Review the test files for implementation examples
- Explore the function implementations for custom functions

---

**Congratulations on completing the GENESIS Core Lib tutorial!** 🎉

You now have the skills and knowledge to effectively generate high-quality synthetic data for your specific needs. Remember to always validate your results, consider privacy implications, and follow best practices for the best outcomes.

Happy synthetic data generation! 🚀
