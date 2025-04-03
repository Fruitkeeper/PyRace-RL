import sys, os
import math, random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # to avoid some "memory" errors with TkAgg backend
import matplotlib.pyplot as plt
from collections import deque
import time

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
VERSION_NAME = 'DQN_v03'  # the name for our enhanced model

REPORT_EPISODES = 500  # report (plot) every...
DISPLAY_EPISODES = 5   # display live game every...

# Visual Settings
DISPLAY_SLEEP = 0.1    # seconds to sleep between steps when displaying
DISPLAY_MODE = True    # Set to False to disable display entirely for faster training

# Training settings
DEMO_MODE = False      # Set to True for a short demo run
DEMO_EPISODES = 100    # Number of episodes to run in demo mode

# Define the enhanced DQN neural network with more capacity
class EnhancedDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EnhancedDQN, self).__init__()
        # Wider network with more neurons to handle continuous input
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# The ReplayBuffer class will store experiences for training
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # Convert state and next_state to numpy arrays with consistent shape
        state_array = np.array(state, dtype=np.float32)
        next_state_array = np.array(next_state, dtype=np.float32)
        self.buffer.append((state_array, action, reward, next_state_array, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        # Ensure state and next_state are properly shaped numpy arrays before converting to tensors
        state_array = np.array(state, dtype=np.float32)
        next_state_array = np.array(next_state, dtype=np.float32)
        
        return (
            torch.FloatTensor(state_array),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state_array),
            torch.FloatTensor(done)
        )
    
    def __len__(self):
        return len(self.buffer)


# The EnhancedDQNAgent class with improved training methods
class EnhancedDQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.0005, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.998, 
                 buffer_size=200000, batch_size=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Initialize networks
        self.policy_net = EnhancedDQN(state_dim, action_dim)
        self.target_net = EnhancedDQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Use Adam optimizer with smaller learning rate
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Larger buffer for more experience storage
        self.memory = ReplayBuffer(buffer_size)
        
        self.steps_done = 0
        self.update_target_every = 5  # Update target network more frequently
        
        # Track losses for monitoring
        self.losses = []
    
    def select_action(self, state, explore=True):
        # Epsilon-greedy action selection
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0  # Return 0 loss if not enough samples
        
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state).gather(1, action.unsqueeze(1))
        
        # Double DQN: select best actions using policy net, but evaluate using target net
        with torch.no_grad():
            next_actions = self.policy_net(next_state).max(1)[1].unsqueeze(1)
            next_state_values = self.target_net(next_state).gather(1, next_actions).squeeze(1)
            
            # Compute the expected Q values
            expected_state_action_values = reward + (1 - done) * self.gamma * next_state_values
        
        # Compute Huber loss (more robust to outliers)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        
        # Track loss
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon (slower decay for better exploration)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss_value
    
    def save(self, file_path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'losses': self.losses
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
            if 'losses' in checkpoint:
                self.losses = checkpoint['losses']
            print(f"Model loaded from {file_path}")
        else:
            print(f"No model found at {file_path}")


def simulate(agent, env, learning=True, episode_start=0, num_episodes=65000, max_t=2000):
    env.set_view(DISPLAY_MODE)  # Set view mode based on global setting
    total_rewards = []
    episode_lengths = []
    max_reward = -10_000
    total_reward = 0  # Initialize to prevent error when loading saved models
    episode_length = 0  # Initialize to prevent error when loading saved models
    
    # Create logs directory if it doesn't exist
    logs_dir = f'logs_{VERSION_NAME}'
    os.makedirs(logs_dir, exist_ok=True)
    
    # Adjust number of episodes for demo mode
    if DEMO_MODE and learning:
        num_episodes = min(num_episodes, DEMO_EPISODES)
        print(f"DEMO MODE: Running {num_episodes} episodes")
    
    for episode in range(episode_start, num_episodes + episode_start):
        # Display training progress in console
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes + episode_start}, Epsilon: {agent.epsilon:.4f}")
            
        if episode > 0:
            total_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            
            # Report and save model
            if learning and episode % REPORT_EPISODES == 0:
                # Plot rewards
                plt.figure(figsize=(12, 8))
                
                plt.subplot(2, 2, 1)
                plt.plot(total_rewards)
                plt.ylabel('Total Reward')
                plt.xlabel('Episodes')
                plt.title(f'Training Rewards (Episode {episode})')
                
                # Plot episode lengths
                plt.subplot(2, 2, 2)
                plt.plot(episode_lengths)
                plt.ylabel('Episode Length')
                plt.xlabel('Episodes')
                plt.title('Steps per Episode')
                
                # Plot loss if available
                if len(agent.losses) > 0:
                    plt.subplot(2, 2, 3)
                    plt.plot(agent.losses[-1000:])  # Plot the last 1000 losses
                    plt.ylabel('Loss')
                    plt.xlabel('Training steps')
                    plt.title('Training Loss (Last 1000 steps)')
                
                # Plot moving average of rewards
                plt.subplot(2, 2, 4)
                window_size = min(50, len(total_rewards))
                moving_avg = np.convolve(np.array(total_rewards), np.ones(window_size)/window_size, mode='valid')
                plt.plot(moving_avg)
                plt.ylabel('Avg Reward')
                plt.xlabel('Episodes')
                plt.title(f'Moving Average ({window_size} episodes)')
                
                plt.tight_layout()
                
                # Save the plot
                plot_file = f'{logs_dir}/rewards_episode_{episode}.png'
                plt.savefig(plot_file)
                print(f"Plot saved to {plot_file}")
                plt.close()
                
                # Save model
                os.makedirs(f'models_{VERSION_NAME}', exist_ok=True)
                model_file = f'models_{VERSION_NAME}/dqn_model_{episode}.pt'
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
        
        for t in range(max_t):
            # Select and perform an action
            action = agent.select_action(obv, explore=learning)
            next_obv, reward, done, _, info = env.step(action)
            env.remember(obv, action, reward, next_obv, done)
            
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
                # Show more detailed information
                msgs = [
                    'SIMULATE',
                    f'Episode: {episode}',
                    f'Time steps: {t}',
                    f'Check: {info["check"]}/{len(check_point)}',
                    f'Dist: {info["dist"]:.1f}',
                    f'Speed: {info["speed"]:.1f}',
                    f'Crash: {info["crash"]}',
                    f'Reward: {total_reward:.0f}',
                    f'Max Reward: {max_reward:.0f}',
                    f'Epsilon: {agent.epsilon:.4f}'
                ]
                
                if learning and num_updates > 0:
                    msgs.append(f'Avg Loss: {episode_loss/num_updates:.4f}')
                    
                env.set_msgs(msgs)
                env.render()
                
                # Add a sleep to slow down display and make it more visible
                time.sleep(DISPLAY_SLEEP)
            
            if done or t >= max_t - 1:
                if total_reward > max_reward:
                    max_reward = total_reward
                # Log episode results to a file
                if episode % 10 == 0:  # Log every 10 episodes
                    avg_loss = episode_loss / max(1, num_updates)
                    with open(f'{logs_dir}/training_log.txt', 'a') as f:
                        f.write(f'Episode {episode}, Steps: {t}, Reward: {total_reward:.0f}, ' +
                               f'Max Reward: {max_reward:.0f}, Epsilon: {agent.epsilon:.4f}, ' +
                               f'Avg Loss: {avg_loss:.4f}\n')
                break


def load_and_play(episode, learning=False):
    # Load a saved model and play the game
    agent = EnhancedDQNAgent(state_dim=len(env.observation_space.low), action_dim=env.action_space.n)
    model_file = f'models_{VERSION_NAME}/dqn_model_{episode}.pt'
    agent.load(model_file)
    
    # Play the game
    simulate(agent, env, learning=learning, episode_start=episode)


if __name__ == "__main__":
    # Initialize the enhanced environment
    env = gym.make("Pyrace-v3").unwrapped  # skip the TimeLimit and OrderEnforcing default wrappers
    print('env', type(env))
    
    # Create directories for saving models and logs
    os.makedirs(f'models_{VERSION_NAME}', exist_ok=True)
    os.makedirs(f'logs_{VERSION_NAME}', exist_ok=True)
    
    # Print environment information
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    
    # Import the check_point definition for display
    from gym_race.envs.pyrace_2d_continuous import check_point
    
    # Hyperparameters - adjusted for better performance with continuous observations
    NUM_EPISODES = 65_000
    MAX_T = 2000
    GAMMA = 0.99            # discount factor
    EPSILON_START = 1.0     # start with 100% exploration
    EPSILON_MIN = 0.01      # minimum exploration rate
    EPSILON_DECAY = 0.998   # slower decay for better exploration with continuous states
    LEARNING_RATE = 0.0005  # smaller learning rate for stability
    BUFFER_SIZE = 200000    # larger replay buffer for more diverse experiences
    BATCH_SIZE = 128        # larger batch size for more stable updates
    
    # Initialize the enhanced DQN agent
    agent = EnhancedDQNAgent(
        state_dim=len(env.observation_space.low),
        action_dim=env.action_space.n,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Run mode selection
    print("Select run mode:")
    print("1. Train a new model")
    print("2. Load and continue training a saved model")
    print("3. Play with a saved model (no learning)")
    print("4. Demo mode (short training run with frequent display)")
    
    mode = input("Enter mode (1-4): ")
    
    if mode == "1":
        # Train a new model
        simulate(agent, env, learning=True, num_episodes=NUM_EPISODES, max_t=MAX_T)
    elif mode == "2":
        # Load and continue training
        episode = int(input("Enter episode number to load: "))
        load_and_play(episode, learning=True)
    elif mode == "3":
        # Play with a saved model
        episode = int(input("Enter episode number to load: "))
        DISPLAY_MODE = True   # Force display on
        DISPLAY_SLEEP = 0.1   # Slower for better viewing
        load_and_play(episode, learning=False)
    elif mode == "4":
        # Demo mode - short training run with more frequent display
        DEMO_MODE = True
        DISPLAY_EPISODES = 1  # Display every episode
        DISPLAY_SLEEP = 0.05  # Slightly faster for training
        simulate(agent, env, learning=True, num_episodes=DEMO_EPISODES, max_t=MAX_T)
    else:
        print("Invalid mode selected. Exiting.") 