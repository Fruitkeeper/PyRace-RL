# PyRace Reinforcement Learning

This repository contains implementations of reinforcement learning algorithms for the PyRace environment, a 2D racing game.

## Implementations

1. **Q-Table Approach**: Traditional RL with discretized state space
   - File: `Pyrace_RL_QTable.py`

2. **Deep Q-Network (DQN) Approach**: Modern deep RL implementation
   - File: `Pyrace_RL_DQN.py`
   - Details: See [DQN README](./DQN_README.md)

## Environment

The custom PyRace environment is a 2D racing game where an agent learns to navigate a track. It's implemented as a custom Gymnasium environment in the `gym_race` package.

## Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Q-Table Implementation

```bash
python Pyrace_RL_QTable.py
```

### Running the DQN Implementation

```bash
python Pyrace_RL_DQN.py
```

You can modify these files to:
- Train a new model
- Continue training from a saved model
- Play using a trained model without further learning

## Model Selection

Both implementations allow for:
1. Training from scratch
2. Loading a saved model and continuing training
3. Loading a saved model to play without learning

See the respective files for instructions on how to switch between these modes.

## Acknowledgements

This project is part of the Reinforcement Learning course. 