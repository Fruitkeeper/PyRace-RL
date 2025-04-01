# Conversion from Q-table to Deep Q-Network (DQN)

This document explains the key differences between the original Q-table implementation (`Pyrace_RL_QTable.py`) and the new Deep Q-Network implementation (`Pyrace_RL_DQN.py`).

## Overview

### Q-table Approach
- Uses a discrete lookup table to store Q-values for each (state, action) pair
- State space must be discretized into buckets
- Limited capacity to generalize across similar states
- Memory-intensive for large state spaces

### DQN Approach
- Uses a neural network to approximate the Q-function
- Can handle continuous state spaces directly
- Better generalization capabilities
- Uses experience replay and target networks for stable learning

## Key Components in DQN Implementation

1. **Neural Network Architecture**
   - A simple feedforward neural network with two hidden layers (128 neurons each)
   - ReLU activation functions for hidden layers
   - Linear output layer representing Q-values for each action

2. **Experience Replay**
   - Stores transitions (state, action, reward, next_state, done) in a replay buffer
   - Randomly samples batches for training to break correlations between consecutive samples
   - Improves stability and data efficiency

3. **Target Network**
   - Separate network with identical architecture used for computing target Q-values
   - Periodically updated to match the policy network
   - Reduces instability during training

4. **Epsilon-Greedy Exploration**
   - Starts with high exploration (epsilon=1.0) and gradually decreases
   - Balances exploration and exploitation during training

## How to Use

### Training a New Model
```python
# The following line is uncommented in the script to train a new model
simulate(agent, env, learning=True, num_episodes=NUM_EPISODES, max_t=MAX_T)
```

### Loading and Continuing Training
```python
# Uncomment this line to load a saved model and continue training
# load_and_play(3500, learning=True)
```

### Playing with a Trained Model (No Learning)
```python
# Uncomment this line to load a saved model and play without learning
# load_and_play(3500, learning=False)
```

## Hyperparameters

- `GAMMA = 0.99` - Discount factor for future rewards
- `EPSILON_START = 1.0` - Initial exploration rate
- `EPSILON_MIN = 0.01` - Minimum exploration rate
- `EPSILON_DECAY = 0.995` - Decay rate for exploration
- `LEARNING_RATE = 0.001` - Learning rate for neural network optimizer
- `BUFFER_SIZE = 100000` - Capacity of the replay buffer
- `BATCH_SIZE = 64` - Number of samples used for each training step

## Dependencies

This implementation requires PyTorch in addition to the original dependencies:
- PyTorch
- NumPy
- Matplotlib
- Gymnasium 