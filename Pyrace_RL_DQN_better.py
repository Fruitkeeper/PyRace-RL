import sys, os
import math, random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # to avoid some "memory" errors with TkAgg backend
import matplotlib.pyplot as plt
from collections import deque
import time
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
import gym_race
"""
This imports the enhanced race environment with continuous observation space and expanded action space
registered as "Pyrace-v3"
"""
VERSION_NAME = 'DQN_v04'  # the name for our improved model

REPORT_EPISODES = 500  # report (plot) every...
DISPLAY_EPISODES = 10   # display live game every...

# Visual Settings
DISPLAY_SLEEP = 0.1    # seconds to sleep between steps when displaying
DISPLAY_MODE = True    # Set to False to disable display entirely for faster training

# Training settings
DEMO_MODE = False      # Set to True for a short demo run
DEMO_EPISODES = 100    # Number of episodes to run in demo mode

# Enable CUDA if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}") 

# Define the improved DQN neural network with optimized architecture
class ImprovedDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImprovedDQN, self).__init__()
        # Deeper network with more neurons for better feature extraction
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout1 = nn.Dropout(0.2)  # Dropout for regularization
        
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(256, 128)
        
        self.fc4 = nn.Linear(128, output_dim)
        
        # Initialize weights with Xavier/Glorot initialization for better training dynamics
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        
    def forward(self, x):
        # Check if input is a single sample or a batch
        if x.dim() == 1:
            # Add batch dimension for a single sample
            x = x.unsqueeze(0)
            
        x = F.leaky_relu(self.fc1(x))  # LeakyReLU for better gradient flow
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)  # No activation on output layer for Q-values 

# Alternative network with Layer Normalization instead of Batch Normalization
class ImprovedDQNWithLayerNorm(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImprovedDQNWithLayerNorm, self).__init__()
        # Deeper network with more neurons for better feature extraction
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)  # Layer normalization works with any batch size
        self.dropout1 = nn.Dropout(0.2)  # Dropout for regularization
        
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)
        
        self.fc4 = nn.Linear(128, output_dim)
        
        # Initialize weights with Xavier/Glorot initialization for better training dynamics
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        
    def forward(self, x):
        # Check if input is a single sample or a batch
        if x.dim() == 1:
            # Add batch dimension for a single sample
            x = x.unsqueeze(0)
            
        x = F.leaky_relu(self.ln1(self.fc1(x)))  # LeakyReLU for better gradient flow
        x = self.dropout1(x)
        x = F.leaky_relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.leaky_relu(self.ln3(self.fc3(x)))
        return self.fc4(x)  # No activation on output layer for Q-values

# An improved ReplayBuffer with prioritized experience replay
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = no prioritization, 1 = full prioritization)
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # For beta calculation
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        # Convert state and next_state to numpy arrays with consistent shape
        state_array = np.array(state, dtype=np.float32)
        next_state_array = np.array(next_state, dtype=np.float32)
        
        # Use max priority for new experiences
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append((state_array, action, reward, next_state_array, done))
            self.size += 1
        else:
            self.buffer[self.position] = (state_array, action, reward, next_state_array, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if self.size < batch_size:
            # Not enough samples yet, return None
            return None
        
        # Calculate beta based on training progress
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        
        # Get probabilities based on priorities
        if self.size == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.size]
        
        # Convert priorities to probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(probs), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize
        weights = np.array(weights, dtype=np.float32)
        
        # Get samples based on indices
        samples = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, done = zip(*samples)
        
        # Ensure state and next_state are properly shaped numpy arrays before converting to tensors
        state_array = np.array(state, dtype=np.float32)
        next_state_array = np.array(next_state, dtype=np.float32)
        
        return (
            torch.FloatTensor(state_array).to(DEVICE),
            torch.LongTensor(action).to(DEVICE),
            torch.FloatTensor(reward).to(DEVICE),
            torch.FloatTensor(next_state_array).to(DEVICE),
            torch.FloatTensor(done).to(DEVICE),
            indices,
            torch.FloatTensor(weights).to(DEVICE)
        )
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return self.size 

# The ImprovedDQNAgent class with advanced training techniques
class ImprovedDQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.0003, gamma=0.99, 
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=10000, 
                 buffer_size=100000, batch_size=64, target_update_freq=1000,
                 use_layer_norm=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_rate = learning_rate
        
        # Initialize networks - use LayerNorm by default as it works better with varying batch sizes
        if use_layer_norm:
            self.policy_net = ImprovedDQNWithLayerNorm(state_dim, action_dim).to(DEVICE)
            self.target_net = ImprovedDQNWithLayerNorm(state_dim, action_dim).to(DEVICE)
        else:
            self.policy_net = ImprovedDQN(state_dim, action_dim).to(DEVICE)
            self.target_net = ImprovedDQN(state_dim, action_dim).to(DEVICE)
            
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set to evaluation mode
        
        # Use Adam optimizer with learning rate scheduler for adaptive learning
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=1000, verbose=True
        )
        
        # Prioritized replay buffer for more efficient learning
        self.memory = PrioritizedReplayBuffer(buffer_size)
        
        self.steps_done = 0
        
        # Track metrics for monitoring
        self.losses = []
        self.avg_q_values = []
        self.exploration_rate = []  # Track epsilon over time
    
    def get_epsilon(self):
        # Linear annealing from epsilon_start to epsilon_min over epsilon_decay steps
        return max(self.epsilon_min, 
                   self.epsilon_start - (self.steps_done * (self.epsilon_start - self.epsilon_min) / self.epsilon_decay))
    
    def select_action(self, state, explore=True):
        # Calculate current epsilon
        epsilon = self.get_epsilon() if explore else 0.0
        
        # Track exploration rate
        if len(self.exploration_rate) < 10000:  # Limit storage to save memory
            self.exploration_rate.append(epsilon)
        
        # Epsilon-greedy action selection
        if explore and random.random() < epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(DEVICE)
            q_values = self.policy_net(state_tensor)
            
            # Track average Q-values
            if len(self.avg_q_values) < 10000:  # Limit storage
                self.avg_q_values.append(q_values.mean().item())
                
            return q_values.max(1)[1].item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        # Increment steps
        self.steps_done += 1
        
        # Only train if we have enough samples
        if len(self.memory) < self.batch_size:
            return 0  # Return 0 loss if not enough samples
        
        # Sample batch with priorities
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return 0
            
        state, action, reward, next_state, done, indices, weights = batch
        
        # Double DQN: select actions using policy network but evaluate with target network
        with torch.no_grad():
            # Get actions from policy network
            next_q_values = self.policy_net(next_state)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            
            # Evaluate using target network
            next_state_values = self.target_net(next_state).gather(1, next_actions).squeeze(1)
            
            # Compute expected Q values
            expected_q_values = reward + (1 - done) * self.gamma * next_state_values
        
        # Compute current Q values
        current_q_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        
        # Compute TD errors for prioritization
        td_errors = torch.abs(current_q_values - expected_q_values).detach().cpu().numpy()
        
        # Compute loss with importance sampling weights
        loss = F.smooth_l1_loss(current_q_values, expected_q_values, reduction='none')
        weighted_loss = (loss * weights).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        
        self.optimizer.step()
        
        # Update learning rate based on loss
        self.scheduler.step(weighted_loss)
        
        # Update priorities in replay buffer
        new_priorities = td_errors + 1e-6  # Small constant to ensure non-zero priorities
        self.memory.update_priorities(indices, new_priorities)
        
        # Update target network periodically
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Track loss
        loss_value = weighted_loss.item()
        if len(self.losses) < 10000:  # Limit storage
            self.losses.append(loss_value)
        
        return loss_value
    
    def save(self, file_path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'steps_done': self.steps_done,
            'losses': self.losses,
            'avg_q_values': self.avg_q_values,
            'exploration_rate': self.exploration_rate
        }, file_path)
        print(f"Model saved to {file_path}")
    
    def load(self, file_path):
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            if 'steps_done' in checkpoint:
                self.steps_done = checkpoint['steps_done']
            if 'losses' in checkpoint:
                self.losses = checkpoint['losses']
            if 'avg_q_values' in checkpoint:
                self.avg_q_values = checkpoint['avg_q_values']
            if 'exploration_rate' in checkpoint:
                self.exploration_rate = checkpoint['exploration_rate']
                
            print(f"Model loaded from {file_path}")
        else:
            print(f"No model found at {file_path}")
            print("Starting with a new model.") 

def simulate(agent, env, learning=True, episode_start=0, num_episodes=50000, max_t=2000):
    """
    Run simulation with advanced metrics tracking and visualization
    
    Args:
        agent: The RL agent to train or evaluate
        env: The environment to interact with
        learning: Whether to train the agent or just evaluate
        episode_start: Episode number to start from (for continued training)
        num_episodes: Total number of episodes to run
        max_t: Maximum steps per episode
    """
    env.set_view(DISPLAY_MODE)  # Set view mode based on global setting
    
    # Initialize metrics
    total_rewards = []
    episode_lengths = []
    max_reward = -10_000
    total_reward = 0
    episode_length = 0
    
    # Create directories for logs and models
    logs_dir = f'logs_{VERSION_NAME}'
    models_dir = f'models_{VERSION_NAME}'
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Adjust number of episodes for demo mode
    if DEMO_MODE and learning:
        num_episodes = min(num_episodes, DEMO_EPISODES)
        print(f"DEMO MODE: Running {num_episodes} episodes")
    
    # Create a CSV file to save metrics
    metrics_file = f'{logs_dir}/training_metrics.csv'
    if episode_start == 0 or not os.path.exists(metrics_file):
        with open(metrics_file, 'w') as f:
            f.write("Episode,Steps,Reward,MaxReward,Epsilon,AvgLoss,AvgQValue\n")
    
    for episode in range(episode_start, num_episodes + episode_start):
        # Display training progress in console
        if episode % 10 == 0:
            epsilon = agent.get_epsilon() if hasattr(agent, 'get_epsilon') else agent.epsilon
            print(f"Episode {episode}/{num_episodes + episode_start}, Epsilon: {epsilon:.4f}")
            
        # Save metrics from previous episode
        if episode > episode_start:
            total_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            
            # Report and save model periodically
            if learning and episode % REPORT_EPISODES == 0:
                # Create advanced visualization with more metrics
                create_advanced_visualization(agent, total_rewards, episode_lengths, episode, logs_dir)
                
                # Save model
                model_file = f'{models_dir}/dqn_model_{episode}.pt'
                agent.save(model_file)
                
                # Save training data as numpy arrays for later analysis
                np.save(f'{logs_dir}/rewards_{episode}.npy', np.array(total_rewards))
                np.save(f'{logs_dir}/episode_lengths_{episode}.npy', np.array(episode_lengths))
        
        # Reset the environment
        obv, _ = env.reset()
        total_reward = 0
        episode_length = 0
        
        # Set display mode for this episode
        should_display = (episode % DISPLAY_EPISODES == 0) and DISPLAY_MODE
        if should_display:
            env.pyrace.mode = 2  # continuous display of game
            print(f"Displaying episode {episode}")
        elif not learning:
            env.pyrace.mode = 2  # Always display when not learning
        
        episode_loss = 0
        num_updates = 0
        episode_q_values = []
        
        for t in range(max_t):
            # Select and perform an action
            action = agent.select_action(obv, explore=learning)
            next_obv, reward, done, _, info = env.step(action)
            
            if learning:
                # Store the transition in memory
                agent.remember(obv, action, reward, next_obv, done)
                
                # Perform one step of the optimization
                loss = agent.replay()
                if loss > 0:
                    episode_loss += loss
                    num_updates += 1
            
            # Move to the next state
            obv = next_obv
            total_reward += reward
            episode_length += 1
            
            # Display the game
            if should_display or (env.pyrace.mode == 2):
                # Get current epsilon in a way that works with both agent implementations
                epsilon = agent.get_epsilon() if hasattr(agent, 'get_epsilon') else agent.epsilon
                
                # Show detailed information
                msgs = [
                    f'SIMULATE {VERSION_NAME}',
                    f'Episode: {episode}',
                    f'Time steps: {t}',
                    f'Check: {info["check"]}/{len(check_point)}',
                    f'Dist: {info["dist"]:.1f}',
                    f'Speed: {info["speed"]:.1f}',
                    f'Crash: {info["crash"]}',
                    f'Reward: {total_reward:.0f}',
                    f'Max Reward: {max_reward:.0f}',
                    f'Epsilon: {epsilon:.4f}'
                ]
                
                if learning and num_updates > 0:
                    msgs.append(f'Avg Loss: {episode_loss/num_updates:.4f}')
                    
                env.set_msgs(msgs)
                env.render()
                
                # Add a sleep to slow down display and make it more visible
                time.sleep(DISPLAY_SLEEP)
            
            if done or t >= max_t - 1:
                # Update max reward
                if total_reward > max_reward:
                    max_reward = total_reward
                
                # Calculate average metrics
                avg_loss = episode_loss / max(1, num_updates)
                avg_q = np.mean(agent.avg_q_values[-100:]) if hasattr(agent, 'avg_q_values') and agent.avg_q_values else 0
                
                # Log episode results to CSV
                with open(metrics_file, 'a') as f:
                    epsilon = agent.get_epsilon() if hasattr(agent, 'get_epsilon') else agent.epsilon
                    f.write(f"{episode},{t},{total_reward:.0f},{max_reward:.0f},{epsilon:.4f},{avg_loss:.4f},{avg_q:.4f}\n")
                
                # Detailed console output every 10 episodes
                if episode % 10 == 0:
                    epsilon = agent.get_epsilon() if hasattr(agent, 'get_epsilon') else agent.epsilon
                    print(f"Episode {episode} - Steps: {t}, Reward: {total_reward:.0f}, "
                          f"Max: {max_reward:.0f}, Epsilon: {epsilon:.4f}, Loss: {avg_loss:.4f}")
                break
    
    # Final save at the end of training
    if learning:
        model_file = f'{models_dir}/dqn_model_final.pt'
        agent.save(model_file)
        print(f"Final model saved to {model_file}")
    
    return total_rewards, episode_lengths


def create_advanced_visualization(agent, total_rewards, episode_lengths, episode, logs_dir):
    """Create advanced visualizations with multiple metrics"""
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(3, 2, 1)
    plt.plot(total_rewards)
    plt.ylabel('Total Reward')
    plt.xlabel('Episodes')
    plt.title(f'Training Rewards (Episode {episode})')
    
    # Plot episode lengths
    plt.subplot(3, 2, 2)
    plt.plot(episode_lengths)
    plt.ylabel('Episode Length')
    plt.xlabel('Episodes')
    plt.title('Steps per Episode')
    
    # Plot loss if available
    if hasattr(agent, 'losses') and len(agent.losses) > 0:
        plt.subplot(3, 2, 3)
        plt.plot(agent.losses[-min(1000, len(agent.losses)):])
        plt.ylabel('Loss')
        plt.xlabel('Training steps')
        plt.title('Training Loss (Recent steps)')
    
    # Plot moving average of rewards
    plt.subplot(3, 2, 4)
    window_size = min(50, len(total_rewards))
    if len(total_rewards) >= window_size:
        moving_avg = np.convolve(np.array(total_rewards), np.ones(window_size)/window_size, mode='valid')
        plt.plot(moving_avg)
        plt.ylabel('Avg Reward')
        plt.xlabel('Episodes')
        plt.title(f'Moving Average ({window_size} episodes)')
    
    # Plot average Q-values if available
    if hasattr(agent, 'avg_q_values') and len(agent.avg_q_values) > 0:
        plt.subplot(3, 2, 5)
        plt.plot(agent.avg_q_values)
        plt.ylabel('Avg Q-Value')
        plt.xlabel('Steps')
        plt.title('Average Q-Values')
    
    # Plot exploration rate if available
    if hasattr(agent, 'exploration_rate') and len(agent.exploration_rate) > 0:
        plt.subplot(3, 2, 6)
        plt.plot(agent.exploration_rate)
        plt.ylabel('Epsilon')
        plt.xlabel('Steps')
        plt.title('Exploration Rate')
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = f'{logs_dir}/metrics_episode_{episode}.png'
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")
    plt.close()


def load_and_play(episode, learning=False):
    """Load a saved model and play/continue training"""
    # Initialize environment
    env = gym.make("Pyrace-v3").unwrapped
    
    # Initialize agent
    agent = ImprovedDQNAgent(
        state_dim=len(env.observation_space.low),
        action_dim=env.action_space.n,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        use_layer_norm=USE_LAYER_NORM
    )
    
    # Load model
    model_file = f'models_{VERSION_NAME}/dqn_model_{episode}.pt'
    agent.load(model_file)
    
    # If not learning, set epsilon to minimum for exploitation
    if not learning:
        agent.steps_done = agent.epsilon_decay  # Force epsilon to minimum
    
    # Play/train
    simulate(agent, env, learning=learning, episode_start=episode)

if __name__ == "__main__":
    # Initialize the enhanced environment
    env = gym.make("Pyrace-v3").unwrapped  # skip the TimeLimit and OrderEnforcing default wrappers
    print('Environment:', type(env))
    
    # Print environment information
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    
    # Import the check_point definition for display
    from gym_race.envs.pyrace_2d_continuous import check_point
    
    # Improved hyperparameters
    NUM_EPISODES = 50_000
    MAX_T = 2000
    
    # Deep Q-Network parameters
    GAMMA = 0.99                    # Discount factor
    EPSILON_START = 1.0             # Initial exploration rate
    EPSILON_MIN = 0.01              # Minimum exploration rate
    EPSILON_DECAY = 100000          # Steps over which to anneal epsilon (linear decay)
    LEARNING_RATE = 0.0003          # Learning rate for optimizer
    BUFFER_SIZE = 100000            # Replay buffer size
    BATCH_SIZE = 64                 # Batch size for training
    TARGET_UPDATE_FREQ = 1000       # How often to update target network
    USE_LAYER_NORM = True           # Use layer normalization (handles single sample batches better)
    
    # Run mode selection
    print("\n========== PyRace Reinforcement Learning ==========")
    print(f"Using model version: {VERSION_NAME}")
    print(f"Using device: {DEVICE}")
    print(f"Using Layer Normalization: {USE_LAYER_NORM}")
    print("\nSelect run mode:")
    print("1. Train a new model from scratch")
    print("2. Load and continue training a saved model")
    print("3. Play with a saved model (no learning)")
    print("4. Demo mode (short training run with frequent display)")
    print("5. Compare performance between episodes")
    
    try:
        mode = input("\nEnter mode (1-5): ")
        
        if mode == "1":
            # Train a new model from scratch
            print(f"\nTraining new model with the following parameters:")
            print(f"Episodes: {NUM_EPISODES}, Max steps: {MAX_T}")
            print(f"Gamma: {GAMMA}, Learning rate: {LEARNING_RATE}")
            print(f"Epsilon: {EPSILON_START} -> {EPSILON_MIN} over {EPSILON_DECAY} steps")
            print(f"Batch size: {BATCH_SIZE}, Buffer size: {BUFFER_SIZE}")
            print(f"Target update frequency: {TARGET_UPDATE_FREQ}")
            print(f"Using Layer Normalization: {USE_LAYER_NORM}")
            
            # Initialize agent
            agent = ImprovedDQNAgent(
                state_dim=len(env.observation_space.low),
                action_dim=env.action_space.n,
                learning_rate=LEARNING_RATE,
                gamma=GAMMA,
                epsilon_start=EPSILON_START,
                epsilon_min=EPSILON_MIN,
                epsilon_decay=EPSILON_DECAY,
                buffer_size=BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                target_update_freq=TARGET_UPDATE_FREQ,
                use_layer_norm=USE_LAYER_NORM
            )
            
            # Train
            simulate(agent, env, learning=True, num_episodes=NUM_EPISODES, max_t=MAX_T)
            
        elif mode == "2":
            # Load and continue training
            episode = int(input("Enter episode number to load: "))
            print(f"\nContinuing training from episode {episode}")
            load_and_play(episode, learning=True)
            
        elif mode == "3":
            # Play with a saved model
            episode = int(input("Enter episode number to load: "))
            print(f"\nPlaying with model from episode {episode}")
            
            # Force display on for better viewing
            DISPLAY_MODE = True
            DISPLAY_SLEEP = 0.05
            load_and_play(episode, learning=False)
            
        elif mode == "4":
            # Demo mode - short training run with more frequent display
            print("\nRunning in demo mode")
            DEMO_MODE = True
            DISPLAY_EPISODES = 1  # Display every episode
            DISPLAY_SLEEP = 0.05  # Slightly faster for training
            
            # Initialize agent
            agent = ImprovedDQNAgent(
                state_dim=len(env.observation_space.low),
                action_dim=env.action_space.n,
                learning_rate=LEARNING_RATE,
                gamma=GAMMA,
                epsilon_start=EPSILON_START,
                epsilon_min=EPSILON_MIN,
                epsilon_decay=EPSILON_DECAY,
                buffer_size=BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                target_update_freq=TARGET_UPDATE_FREQ,
                use_layer_norm=USE_LAYER_NORM
            )
            
            # Run demo
            simulate(agent, env, learning=True, num_episodes=DEMO_EPISODES, max_t=MAX_T)
            
        elif mode == "5":
            # Compare performance between two saved models
            ep1 = int(input("Enter first episode number to load: "))
            ep2 = int(input("Enter second episode number to load: "))
            num_eval_episodes = 10
            
            print(f"\nComparing models from episodes {ep1} and {ep2}")
            print(f"Running {num_eval_episodes} evaluation episodes for each model")
            
            # Force display on but faster
            DISPLAY_MODE = True
            DISPLAY_SLEEP = 0.02
            
            # Set up environment and load models
            env1 = gym.make("Pyrace-v3").unwrapped
            env2 = gym.make("Pyrace-v3").unwrapped
            
            # Create agents
            agent1 = ImprovedDQNAgent(
                state_dim=len(env1.observation_space.low), 
                action_dim=env1.action_space.n,
                use_layer_norm=USE_LAYER_NORM
            )
            agent2 = ImprovedDQNAgent(
                state_dim=len(env2.observation_space.low), 
                action_dim=env2.action_space.n,
                use_layer_norm=USE_LAYER_NORM
            )
            
            # Load models
            agent1.load(f'models_{VERSION_NAME}/dqn_model_{ep1}.pt')
            agent2.load(f'models_{VERSION_NAME}/dqn_model_{ep2}.pt')
            
            # Set epsilon to minimum for pure exploitation
            agent1.steps_done = agent1.epsilon_decay
            agent2.steps_done = agent2.epsilon_decay
            
            # Run evaluation for each model
            print(f"\nEvaluating model from episode {ep1}...")
            rewards1 = []
            for i in range(num_eval_episodes):
                print(f"Evaluation episode {i+1}/{num_eval_episodes}")
                rewards, _ = simulate(agent1, env1, learning=False, num_episodes=1, max_t=MAX_T)
                rewards1.extend(rewards)
            
            print(f"\nEvaluating model from episode {ep2}...")
            rewards2 = []
            for i in range(num_eval_episodes):
                print(f"Evaluation episode {i+1}/{num_eval_episodes}")
                rewards, _ = simulate(agent2, env2, learning=False, num_episodes=1, max_t=MAX_T)
                rewards2.extend(rewards)
            
            # Compare results
            print("\n===== Comparison Results =====")
            print(f"Model from episode {ep1}: Avg reward = {np.mean(rewards1):.0f}, Max = {np.max(rewards1):.0f}")
            print(f"Model from episode {ep2}: Avg reward = {np.mean(rewards2):.0f}, Max = {np.max(rewards2):.0f}")
            
            if np.mean(rewards1) > np.mean(rewards2):
                print(f"Model from episode {ep1} performs better by {np.mean(rewards1) - np.mean(rewards2):.0f} points")
            else:
                print(f"Model from episode {ep2} performs better by {np.mean(rewards2) - np.mean(rewards1):.0f} points")
        
        else:
            print("Invalid mode selected. Exiting.")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc() 