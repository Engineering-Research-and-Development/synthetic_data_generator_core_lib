# GENESIS Core Lib

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-AGPLv3-blue.svg)](LICENSE)


> 🧬 **Advanced Synthetic Data Generation Library** for Python 3.12+

GENESIS Core Lib is a powerful, extensible library for generating high-quality synthetic data using state-of-the-art machine learning models. Perfect for data augmentation, privacy preservation, and ML model testing.

## ✨ Key Features

- 🎯 **Multiple Model Types**: VAEs (TabularVAE, TimeSeriesVAE) and CTGAN
- 📊 **Data Type Support**: Tabular data, time series with group_index, and custom datasets
- 🔧 **Function-Based Generation**: Mathematical functions for controlled data generation
- 📈 **Quality Evaluation**: Built-in metrics for data quality assessment
- 🚀 **High Performance**: Optimized for both CPU and GPU processing
- 🔒 **Privacy Focused**: Designed with privacy preservation in mind

## 🛠️ Installation

### Quick Install
```bash
pip install sdg-core-lib
```

### Development Install
```bash
git clone https://github.com/emiliocimino/generator_core_lib.git
cd generator_core_lib
pip install -e ".[dev]"
```

## 🚀 Quick Start

```python
from sdg_core_lib import Job

# Load your dataset configuration
import json
with open('config.json', 'r') as f:
    config = json.load(f)

# Create and run a synthetic data generation job
job = Job(
    n_rows=config["n_rows"],
    model_info=config["model"],
    dataset=config["dataset"],
    save_filepath=config.get("save_filepath", "./models")
)

# Generate synthetic data
results, metrics, model, schema = job.train()
print(f"Generated {len(results)} synthetic rows")
print(f"Quality metrics: {metrics}")
```

📖 **See [Quick Start Guide](docs/quick-start.md) for detailed examples**

## 🔧 Function-Based Generation

```python
# Generate data using mathematical functions
functions = [
    {
        "feature": "linear_data",
        "function_name": "LinearFunction",
        "parameters": {
            "m": 2.0,
            "q": 1.0,
            "min_value": 0.0,
            "max_value": 100.0
        }
    }
]

job = Job(n_rows=100, functions=functions)
synthetic_data = job.generate_from_functions()
```

## 📚 Documentation

### 📖 [User Documentation](docs/user-documentation.md)
Complete guide for users including:
- Core concepts and architecture
- Data types (tabular, time series, custom)
- Model configurations (VAEs, CTGAN)
- API reference and examples
- Best practices and troubleshooting

### 🔧 [Developer Documentation](docs/developer-documentation.md)
Technical documentation for developers:
- Architecture overview and design patterns
- Extension points and customization
- Development setup and testing
- Code organization and standards

### ⚡ [Quick Start Guide](docs/quick-start.md)
Get started immediately with:
- Installation instructions
- Basic examples and tutorials
- Common use cases
- Troubleshooting tips

### 📋 [Step-by-Step Tutorial](docs/step-by-step-tutorial.md)
Hands-on tutorial covering:
- Complete project workflow
- Real-world examples
- Advanced techniques
- Performance optimization


## 🏗️ Architecture

GENESIS Core Lib follows a modular architecture:

- **Data Generator**: ML models (TabularVAE, TimeSeriesVAE, CTGAN)
- **Dataset**: Data abstraction (Table, TimeSeries) with proper column structure
- **Preprocess**: Data transformation and normalization strategies
- **Postprocess**: Function application and data modification
- **Evaluate**: Quality assessment and statistical metrics

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone repository
git clone https://github.com/emiliocimino/generator_core_lib.git
cd generator_core_lib

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sdg_core_lib

# Run specific test file
pytest tests/test_job.py
```

## 📄 License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with TensorFlow and Keras for deep learning models
- Statistical evaluation using scipy and numpy
- Inspired by state-of-the-art synthetic data generation research

## 📞 Support

- 📖 [Documentation](https://github.com/emiliocimino/generator_core_lib/docs)
- 🐛 [Issues](https://github.com/emiliocimino/generator_core_lib/issues)
- 💬 [Discussions](https://github.com/emiliocimino/generator_core_lib/discussions)

---

**GENESIS Core Lib** - *Generating Tomorrow's Data, Today* 🚀
