import sys, os
import math, random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # to avoid some "memory" errors with TkAgg backend
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
import gym_race
"""
This imports race_env.py (a gym env) and pyrace_2d.py (the race game) and registers the env as "Pyrace-v1"

register(
    id='Pyrace-v1',
    entry_point='gym_race.envs:RaceEnv',
    max_episode_steps=2_000,
)
"""

VERSION_NAME = 'DQN_v01'  # the name for our model
MODEL_DIR = f'models_{VERSION_NAME}'  # Consistent model folder

REPORT_EPISODES = 500  # report (plot) every...
DISPLAY_EPISODES = 100  # display live game every...

# Define the DQN neural network with enhanced capacity and dropout
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # Increased capacity
        self.dropout1 = nn.Dropout(0.2)         # Added dropout to reduce overfitting
        self.fc2 = nn.Linear(256, 256)          # Increased capacity
        self.dropout2 = nn.Dropout(0.2)         # Added dropout
        self.fc3 = nn.Linear(256, 128)          # Extra hidden layer for complexity
        self.fc4 = nn.Linear(128, output_dim)   # Output layer
       
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# The ReplayBuffer class to store experiences
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done)
        )
    
    def __len__(self):
        return len(self.buffer)


# The DQNAgent class handles training and action selection.
class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Initialize networks
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        
        self.steps_done = 0
        self.update_target_every = 10
        self.episodes_done = 0  # Track episodes for epsilon decay
        self.reward_scaling = 0.01  # Scale rewards to smaller range
    
    def select_action(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()
    
    def remember(self, state, action, reward, next_state, done):
        # Scale the reward before storing
        scaled_reward = reward * self.reward_scaling
        self.memory.push(state, action, scaled_reward, next_state, done)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state).gather(1, action.unsqueeze(1))
        
        # Compute max Q(s_{t+1}, a) for all next states from target network
        next_state_values = self.target_net(next_state).max(1)[0].detach()
        
        # Compute expected Q values using the Bellman equation
        expected_state_action_values = reward + (1 - done) * self.gamma * next_state_values
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # Gradient clipping
        self.optimizer.step()
        
        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon after each episode instead of each replay step
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, file_path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, file_path)
        print(f"Model saved to {file_path}")
    
    def load(self, file_path):
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.steps_done = checkpoint['steps_done']
            print(f"Model loaded from {file_path}")
        else:
            print(f"No model found at {file_path}")


def simulate(agent, env, learning=True, episode_start=0, num_episodes=10000, max_t=2000):
    """
    Simulate episodes of the environment with the agent.
    """
    env.set_view(True)  # Enable visualization
    episode_rewards = []
    episode_lengths = []
    best_reward = float('-inf')
    
    for episode in range(episode_start, episode_start + num_episodes):
        state, _ = env.reset()
        total_reward = 0
        t = 0
        
        # Always show the game during training
        env.pyrace.mode = 2  # continuous display of game
        
        while t < max_t:
            action = agent.select_action(state, explore=learning)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if learning:
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
            
            state = next_state
            total_reward += reward
            t += 1
            
            # Display game information
            env.set_msgs([
                f'Episode: {episode}',
                f'Time steps: {t}',
                f'Reward: {total_reward:.0f}',
                f'Epsilon: {agent.epsilon:.4f}',
                f'Best Reward: {best_reward:.0f}'
            ])
            env.render()  # Make sure to render each step
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(t)
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Average Reward (last 10): {avg_reward:.2f}")
            print(f"Average Length (last 10): {avg_length:.2f}")
            print(f"Epsilon: {agent.epsilon:.4f}")
            print("-" * 50)
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(f'models_DQN_v01/best_model_{VERSION_NAME}.pth')
        
        # Save checkpoint every 100 episodes
        if (episode + 1) % 100 == 0:
            agent.save(f'models_DQN_v01/checkpoint_{VERSION_NAME}_episode_{episode+1}.pth')
    
    return episode_rewards, episode_lengths


def load_and_play(episode=None, learning=False):
    """
    Load a saved model and play the game.
    If episode is None, load the best model.
    """
    agent = DQNAgent(state_dim=len(env.observation_space.low), action_dim=4)
    
    # Use best model if no specific episode is provided
    if episode is None:
        model_file = os.path.join(MODEL_DIR, f'best_model_{VERSION_NAME}.pth')
    else:
        model_file = os.path.join(MODEL_DIR, f'checkpoint_{VERSION_NAME}_episode_{episode}.pth')
    
    agent.load(model_file)
    
    # Set network to evaluation mode when not learning
    if not learning:
        agent.policy_net.eval()
    
    # Play the game without further learning
    simulate(agent, env, learning=learning, episode_start=0, num_episodes=10)


if __name__ == "__main__":
    # Initialize the environment
    env = gym.make("Pyrace-v1").unwrapped  # skip the TimeLimit and OrderEnforcing default wrappers
    print('env', type(env))
    
    # Create directories for saving models
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # Print environment information
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    
    # Hyperparameters - Adjusted for better learning
    NUM_EPISODES = 65000  # Increased training duration
    MAX_T = 2000
    GAMMA = 0.99            # Discount factor
    EPSILON_START = 1.0     # Start with 100% exploration
    EPSILON_MIN = 0.01      # Minimum exploration rate
    EPSILON_DECAY = 0.999   # Slower decay rate for exploration
    LEARNING_RATE = 0.0005  # Reduced learning rate for more stable learning
    BUFFER_SIZE = 50000     # Increased buffer size for more diverse experiences
    BATCH_SIZE = 128        # Increased batch size for better gradient estimates
    
    # Initialize the DQN agent with 4 actions (accelerate, brake, turn left, turn right)
    agent = DQNAgent(
        state_dim=len(env.observation_space.low),
        action_dim=4,  # Changed from env.action_space.n to 4
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Train the agent
    print("Starting training...")
    print("You should see a window with the car simulation")
    print("The car will start with random movements and gradually learn to drive")
    print("Press Ctrl+C to stop training at any time")
    
    try:
        episode_rewards, episode_lengths = simulate(agent, env, learning=True, num_episodes=NUM_EPISODES, max_t=MAX_T)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving current model...")
        agent.save(f'{MODEL_DIR}/interrupted_model_{VERSION_NAME}.pth')
    
    # Alternatively, load and continue training a saved model:
    # load_and_play(episode=350, learning=True)
    
    # Or just play with a saved model without learning:
    # load_and_play(episode=None, learning=False)
