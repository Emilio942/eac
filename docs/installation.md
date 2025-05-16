# Installation and Setup

This guide provides step-by-step instructions for installing and setting up the Emergent Adaptive Core (EAC) framework.

## Prerequisites

The EAC framework requires:
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd eac
```

### 2. Create a Virtual Environment (recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install the package in development mode
pip install -e .
```

This will install all the required dependencies specified in `requirements.txt`.

## Configuration

The EAC framework can be configured using YAML configuration files. A default configuration file is provided at `config/default_config.yaml`.

To use a custom configuration, create a copy of the default configuration file and modify the parameters as needed:

```bash
cp config/default_config.yaml config/my_config.yaml
```

Then, when initializing the EAC system, provide the path to your custom configuration:

```python
from eac.main import EmergentAdaptiveCore

# Initialize with custom configuration
eac = EmergentAdaptiveCore(config_path="config/my_config.yaml")
```

## Running Examples

The `examples` directory contains script examples for running the EAC system:

```bash
# Run the basic example
python examples/run_example.py

# Run with custom parameters
python examples/run_example.py --iterations 2000 --complexity 0.7 --dynamics-rate 0.02
```

### Available Command-Line Options

- `--iterations`: Number of iterations to run (default: 1000)
- `--complexity`: Environment complexity (0-1) (default: 0.5)
- `--dynamics-rate`: Environment dynamics rate (0-1) (default: 0.01)
- `--log-interval`: Interval for logging metrics (default: 100)
- `--output-dir`: Directory to save results (default: ./results)

## Development Setup

If you plan to contribute to the EAC framework, you should install the development dependencies:

```bash
pip install -e ".[dev]"
```

This will install additional tools for testing, linting, and documentation.

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'eac'**:
   - Ensure you've installed the package with `pip install -e .`
   - Check that your virtual environment is activated

2. **Missing dependencies**:
   - Run `pip install -r requirements.txt` to ensure all dependencies are installed

3. **Configuration errors**:
   - Validate your YAML configuration file format
   - Ensure all required parameters are present

### Getting Help

If you encounter problems not covered in this guide, please:
1. Check the documentation in the `docs` directory
2. Look for similar issues in the issue tracker
3. Create a new issue with a detailed description of your problem

## Next Steps

Once you have the EAC framework installed and running, you can:
- Explore the architecture by reading the code and documentation
- Experiment with different configuration settings
- Create custom environments to test the system
- Extend the system with new modules or capabilities
