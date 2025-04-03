# PyRace RL - Run Best Model

This script allows you to run and evaluate our group's best trained reinforcement learning model for the PyRace racing environment.

## Features

- Runs the trained model in the PyRace-v3 environment
- Supports multiple evaluation runs for statistical analysis
- Displays real-time performance metrics during execution
- Configurable display speed and model selection

## Usage

Basic usage:

```bash
python run_best_model.py
```

This will run the default model (DQN_v04b, episode 1100) for a single episode.

### Command-line Options

| Option | Description |
|--------|-------------|
| `--version` | Model version name (default: DQN_v04b) |
| `--episode` | Episode number to load (default: 1100) |
| `--runs` | Number of episodes to run (default: 1) |
| `--sleep` | Sleep time between frames (default: 0.05) |

### Examples

Run a specific model:
```bash
python run_best_model.py --version DQN_v04b --episode 1000
```

Run multiple episodes for statistical reliability:
```bash
python run_best_model.py --runs 10
```

Adjust display speed (higher value = slower):
```bash
python run_best_model.py --sleep 0.1
```

## Output

The script provides:
- Real-time display of the racing environment
- Per-episode performance details
- Summary statistics after all runs are complete including:
  - Success rate (completed all checkpoints)
  - Crash rate
  - Timeout rate
  - Average reward, steps, and checkpoints completed

## Requirements

- Python 3.6+
- PyTorch
- Gymnasium
- Pygame
- NumPy
- gym_race (PyRace custom environment) 