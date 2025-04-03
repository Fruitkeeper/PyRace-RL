# Reinforcement Learning Assignment 17 - Laura Cuellar, Luis Gomez, Daniel Sanchez

## Key Improvements in Our Approach

Starting with the provided Q-table implementation, we transitioned to a Deep Q-Network (DQN) approach to overcome the limitations of traditional tabular methods. Here's a breakdown of the key differences and why we made these changes:

### 1. Handling Continuous State Spaces

The biggest limitation we observed in the provided Q-table implementation was its inability to effectively handle continuous state spaces. In the original `Pyrace_RL_QTable.py`, the environment state had to be discretized using the `state_to_bucket` function:

```python
def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        # Discretize continuous values into buckets
        # ...
    return tuple(bucket_indice)
```

This discretization created a coarse representation with only 11 buckets per dimension, resulting in significant information loss. We decided to transition to a neural network approach in our `Pyrace_RL_DQN.py` because it could directly process continuous state values without discretization, enabling much finer-grained decision making.

### 2. Scalability and Dimensionality

In the provided Q-table approach, we noticed the inherent "curse of dimensionality" problem. The Q-table had shape `(11, 11, 11, 11, 11, 3)`, which was already pushing memory limits with just 5 state dimensions. We realized that when we wanted to add more sensor inputs for better environmental awareness, this approach would become impractical.

By implementing a DQN with our `ImprovedDQN` and `ImprovedDQNWithLayerNorm` classes, we could handle much higher-dimensional state spaces efficiently. This allowed us to enrich the agent's perception without the exponential memory growth that a Q-table would require.

### 3. Enhanced Environment for Better Learning

One of the most critical improvements we made was enhancing the environment itself. We transitioned from the basic `race_env.py` to our improved `race_env_continuous.py`. Here are the key differences:

```python
# Original environment (race_env.py)
self.action_space = spaces.Discrete(3)
self.observation_space = spaces.Box(
    np.array([0, 0, 0, 0, 0]), 
    np.array([10, 10, 10, 10, 10]), 
    dtype=int
)

# Our enhanced environment (race_env_continuous.py)
self.action_space = spaces.Discrete(4)  # Added braking action
self.observation_space = spaces.Box(
    np.array([0, 0, 0, 0, 0, 0, 0]), 
    np.array([200, 200, 200, 200, 200, 10, 360]), 
    dtype=np.float32  # Changed to float32 for continuous values
)
```

We made several critical enhancements to the environment:

- **Expanded action space**: We added a braking action (4 actions total instead of 3), giving the agent more control over the car's movement.
  
- **Continuous observation space**: We changed from discrete integer observations to continuous float values, preserving all the nuance in the sensor readings.
  
- **Raw sensor readings**: Instead of bucketing distances into 0-10 ranges, we used the actual pixel distances (0-200), giving the agent much more precise information about its surroundings.
  
- **Added speed and steering angle**: We enhanced the state with the car's current speed and steering angle, allowing the agent to make more informed decisions based on its current dynamics.
  
- **Richer information feedback**: We expanded the info dictionary returned by the step function to include:
  ```python
  info = {
      'dist': self.pyrace.car.distance,
      'check': self.pyrace.car.current_check,
      'crash': not self.pyrace.car.is_alive,
      'speed': self.pyrace.car.speed,
      'angle': self.pyrace.car.angle,
      'time': self.pyrace.car.time_spent
  }
  ```
  
This enhanced environment was crucial for our DQN implementation's success. The continuous state space perfectly complemented the neural network's ability to process continuous inputs, and the additional action (braking) allowed for more nuanced driving strategies.

### 4. Advanced Neural Network Architecture

We designed our DQN with several modern techniques to improve learning:

```python
class ImprovedDQNWithLayerNorm(nn.Module):
    def __init__(self, input_dim, output_dim):
        # ...
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(0.2)
        # ...
```

We decided to use:
- Larger network (256-256-128 neurons) for better feature extraction
- Layer normalization to stabilize training with varying batch sizes
- Dropout for regularization to prevent overfitting
- LeakyReLU activations for better gradient flow
- Xavier/Glorot initialization for improved convergence

These choices dramatically improved our agent's ability to learn complex driving behaviors compared to the simple value-lookup of the provided Q-table implementation.

### 5. Prioritized Experience Replay

One of our most significant innovations was implementing prioritized experience replay:

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        # ...
```

Rather than uniformly sampling past experiences (which the Q-table approach effectively did), we prioritized experiences with higher TD errors. This meant our agent learned more efficiently from surprising or uncommon situations - like crashes or successful checkpoint completions - rather than wasting learning cycles on common, well-understood states.

### 6. Advanced Training Techniques

We implemented several techniques to enhance training stability and efficiency:

- **Double DQN**: We used separate policy and target networks to reduce overestimation bias
- **Adaptive learning rate**: Our implementation uses a learning rate scheduler that adjusts based on performance
- **Epsilon annealing**: We designed a more sophisticated exploration strategy that gradually transitions from exploration to exploitation

### 7. Better Metrics and Visualization

In the provided Q-table implementation, we had very basic reporting. Our DQN version includes comprehensive metrics tracking:

```python
def create_advanced_visualization(agent, total_rewards, episode_lengths, episode, logs_dir):
    # Multiple plots showing rewards, losses, Q-values, etc.
    # ...
```

We integrated visualization of rewards, losses, Q-values, episode lengths, and exploration rates, allowing us to better understand the learning process and diagnose issues.

We also implemented a robust logging system that automatically creates and maintains organized log directories:

```python
# Create directories for logs and models
logs_dir = f'logs_{VERSION_NAME}'
models_dir = f'models_{VERSION_NAME}'
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Create a CSV file to save metrics
metrics_file = f'{logs_dir}/training_metrics.csv'
if episode_start == 0 or not os.path.exists(metrics_file):
    with open(metrics_file, 'w') as f:
        f.write("Episode,Steps,Reward,MaxReward,Epsilon,AvgLoss,AvgQValue\n")
```

This structured logging approach allowed us to:
- Save training metrics to CSV files for easy analysis in external tools
- Automatically generate visualizations at regular intervals
- Store model checkpoints systematically to track progress over time
- Save raw training data as NumPy arrays for detailed post-training analysis

Having this data readily available made it much easier to identify training issues, compare different model versions, and select the best-performing model for deployment.

### 8. Flexible Architecture for Experimentation

We structured our DQN code to make experimentation easier:

```python
if __name__ == "__main__":
    # ...
    print("\nSelect run mode:")
    print("1. Train a new model from scratch")
    print("2. Load and continue training a saved model")
    # ...
```

This modular approach with multiple run modes allowed us to easily train new models, continue training existing ones, evaluate performance, and compare different models - accelerating our research cycle considerably.

### 9. Iterative Development Process and Challenges Overcome

Our path to the final model was not straightforward but involved an iterative process of experimentation and refinement. We went through several model versions before arriving at our best-performing agent:

1. **Early Iterations (v01-v03)**: Our initial DQN implementations showed promise but struggled with consistent performance. The agents would sometimes complete the track but often crashed in challenging corners.

2. **Version v04 - Breaking Through**: With v04, we saw dramatic improvements. The agent was training well and learning to navigate most of the track successfully. However, we noticed a concerning pattern around episode 1200: after performing exceptionally well up to that point, the agent would begin to decline in performance.

3. **Identifying the Speed-Crash Problem**: Through careful analysis of our logs and visualizations, we discovered the root cause of this performance decline. The agent had learned to drive at increasingly high speeds to maximize rewards, but this led to inevitable crashes at a particularly challenging curve in the track. The agent was essentially becoming "overconfident" in its abilities.

4. **Targeted Environment Refinement**: To address this specific issue, we made critical adjustments to the environment's reward function:

   ```python
   # Modified reward function to penalize excessive speed and high-speed crashes
   def evaluate(self):
       # Original reward components
       reward = self.car.reward
       
       # New: quadratic penalty for crashes based on speed
       if not self.car.is_alive:
           speed_factor = self.car.speed / self.car.max_speed
           crash_penalty = -500 * (speed_factor ** 2)  # Harsher penalties for high-speed crashes
           reward += crash_penalty
       
       # New: reward structure that encourages moderate speeds
       if self.car.is_alive:
           # Optimal speed zone with highest rewards
           optimal_speed_min = 0.5 * self.car.max_speed
           optimal_speed_max = 0.8 * self.car.max_speed
           
           if optimal_speed_min <= self.car.speed <= optimal_speed_max:
               reward += 10  # Bonus for maintaining optimal speed
           elif self.car.speed > optimal_speed_max:
               # Gradually reducing reward for excessive speed
               excess = (self.car.speed - optimal_speed_max) / (self.car.max_speed - optimal_speed_max)
               reward -= 20 * excess  # Penalty proportional to how much speed exceeds optimal
               
       return reward
   ```

5. **Version v04b - The Refined Solution**: Instead of starting from scratch, we took the promising model from v04 at episode 1000 (just before the performance decline began) and continued training it with our improved reward structure. This new version, v04b, quickly learned to balance speed with safety, successfully navigating the challenging curve that had been causing crashes.

This iterative process taught us an important lesson in reinforcement learning: sometimes the agent finds unexpected ways to maximize rewards that don't align with our actual goals. By carefully analyzing performance patterns and refining the reward structure to better align with desired behavior, we were able to guide the agent toward truly optimal driving strategies.

## Results and Impact

The transition from the provided Q-table implementation to our DQN approach, combined with our enhanced continuous environment, dramatically improved our agent's performance. Our best DQN model (v04b, episode 1100) successfully navigates the entire track with smooth driving behaviors and strategic speed control around corners. The original Q-table version, while functional as a starting point, never achieved the same level of driving finesse due to its inherent limitations in handling continuous state spaces.

By embracing deep reinforcement learning techniques and designing a more suitable environment, we were able to create an agent that not only completes the course but does so with driving behaviors that more closely resemble human-like control strategies.

## How to Use Our Implementation

### Running Our Best Model

We've created a dedicated script (`run_best_model.py`) that provides a streamlined way to run our best-performing model:

```bash
python run_best_model.py
```

This loads our best model (DQN_v04b, episode 1100) and runs it in the environment with visualization enabled.

For more detailed options and usage examples, please refer to the `BestModel_README.md` file, which contains comprehensive documentation of:
- Command-line options for customizing runs
- Examples for different use cases
- Explanation of the output statistics

Our best model consistently completes the track with smooth driving patterns, demonstrating the effectiveness of our approach in creating a capable racing agent.

### Training with Pyrace_RL_DQN.py

To use our training script and experiment with different models, follow these steps:

1. **Run the training script**:
   ```bash
   python Pyrace_RL_DQN.py
   ```

2. **Select a run mode from the menu**:
   ```
   ========== PyRace Reinforcement Learning ==========
   Using model version: DQN_v04b
   Using device: cpu
   Using Layer Normalization: True

   Select run mode:
   1. Train a new model from scratch
   2. Load and continue training a saved model
   3. Play with a saved model (no learning)
   4. Demo mode (short training run with frequent display)
   5. Compare performance between episodes
   ```

3. **Training options**:
   - **Option 1**: Trains a new model from scratch with our optimal hyperparameters
   - **Option 2**: Continues training from a saved checkpoint (you'll be prompted for the episode number)
   - **Option 3**: Evaluates a saved model without further training
   - **Option 4**: Runs a short demo with frequent visualization for quick experimentation
   - **Option 5**: Compares two saved models head-to-head (you'll be prompted for two episode numbers)

During training, the script automatically saves models after each episode in the `models_DQN_v04b` directory and logs metrics to `logs_DQN_v04b`.

### Project Dependencies

To run our implementation, you'll need the following dependencies:

#### Python Version
- Python 3.6 or higher

#### Required Packages
```
pip install -r requirements.txt
```

Or install the following packages individually:
- PyTorch (1.8.0 or higher): `pip install torch`
- Gymnasium (0.28.1 or higher): `pip install gymnasium`
- NumPy (1.19.0 or higher): `pip install numpy`
- Matplotlib (3.3.0 or higher): `pip install matplotlib`
- Pygame (2.1.0 or higher, for visualization): `pip install pygame`



