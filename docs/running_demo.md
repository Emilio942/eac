# Running the EAC Demo

This guide provides instructions for running the Emergent Adaptive Core (EAC) demonstration.

## Setup

First, make sure you have installed the EAC package as described in the installation guide:

```bash
# Install the package in development mode
pip install -e .
```

## Running the Example

The example script demonstrates the EAC system interacting with a simple environment:

```bash
# Run with default parameters
python examples/run_example.py

# Run with custom parameters
python examples/run_example.py --iterations 500 --complexity 0.7 --dynamics-rate 0.02
```

### Command-Line Options

You can customize the example run with these options:

- `--iterations`: Number of iterations to run (default: 1000)
- `--complexity`: Environment complexity from 0-1 (default: 0.5)
- `--dynamics-rate`: Rate at which the environment changes (default: 0.01)
- `--log-interval`: How often to log metrics (default: 100)
- `--output-dir`: Directory to save results (default: ./results)

## Expected Output

The example will display progress information and key metrics during execution:

```
Starting EAC run with 1000 iterations
Environment complexity: 0.5, dynamics rate: 0.01

Iteration 0/1000 (0.0%)
  Success rate: 0.00
  Modifications: 0
  New abstractions: 0
  Exploration actions: 0

Iteration 100/1000 (10.0%)
  Success rate: 0.23
  Modifications: 3
  New abstractions: 45
  Exploration actions: 28

...

Iteration 900/1000 (90.0%)
  Success rate: 0.65
  Modifications: 15
  New abstractions: 187
  Exploration actions: 272

Completed 1000 iterations in 123.45 seconds
Results saved to ./results
```

The system will also log significant events, such as:

```
Iteration 42: Applied architecture modification: add symbolic_reasoning
Iteration 78: Exploration discovered: pattern_nonlinear
Iteration 156: Successfully solved classification problem
```

## Analyzing Results

After the run completes, results will be saved to the output directory:

1. **Metrics Plots**: `results/metrics_plots.png` contains graphs of key metrics
2. **Metrics Data**: `results/metrics.json` contains raw metric data
3. **EAC Logs**: `results/eac_logs.json` contains detailed event logs
4. **EAC Metrics**: `results/eac_metrics.json` contains detailed performance metrics
5. **Environment Metrics**: `results/environment_metrics.json` contains environment performance data

## What to Look For

When analyzing the results, pay attention to these key indicators:

1. **Success Rate**: Does the system's problem-solving ability improve over time?
2. **Architecture Modifications**: Does the system make more modifications early and then stabilize?
3. **New Abstractions**: Does the rate of new abstractions correlate with improved performance?
4. **Exploration Actions**: Does the system's exploration strategy evolve over time?

## Changing Parameters

Try running the example with different parameters to observe how the system adapts:

1. **High Complexity**: `--complexity 0.9` creates more difficult problems
2. **Dynamic Environment**: `--dynamics-rate 0.05` creates a rapidly changing environment
3. **Long Run**: `--iterations 5000` allows more time for adaptation

## Next Steps

After running the basic example, you can:

1. Modify the environment in `src/eac/environments/simple_environment.py`
2. Experiment with different configuration settings in `config/default_config.yaml`
3. Extend the system with new components or capabilities
