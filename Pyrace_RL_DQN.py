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
this imports race_env.py (a gym env) and pyrace_2d.py (the race game) and registers the env as "Pyrace-v1"

register(
    id='Pyrace-v1',
    entry_point='gym_race.envs:RaceEnv',
    max_episode_steps=2_000,
)
"""
VERSION_NAME = 'DQN_v01'  # the name for our model

REPORT_EPISODES = 500  # report (plot) every...
DISPLAY_EPISODES = 100  # display live game every...


# Define the DQN neural network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# The ReplayBuffer class will store experiences for training
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


# The DQNAgent class will handle training and action selection
class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, buffer_size=10000, batch_size=64):
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
        self.update_target_every = 10  # Update target network every 10 steps
    
    def select_action(self, state, explore=True):
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
            return
        
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state).gather(1, action.unsqueeze(1))
        
        # Compute max Q(s_{t+1}, a) for all next states
        next_state_values = self.target_net(next_state).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_state_action_values = reward + (1 - done) * self.gamma * next_state_values
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # Clip gradients
        self.optimizer.step()
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
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


def simulate(agent, env, learning=True, episode_start=0, num_episodes=65000, max_t=2000):
    env.set_view(True)
    total_rewards = []
    max_reward = -10_000
    
    for episode in range(episode_start, num_episodes + episode_start):
        if episode > 0:
            total_rewards.append(total_reward)
            
            # Report and save model
            if learning and episode % REPORT_EPISODES == 0:
                plt.plot(total_rewards)
                plt.ylabel('rewards')
                plt.show(block=False)
                plt.pause(4.0)
                
                # Save model
                os.makedirs(f'models_{VERSION_NAME}', exist_ok=True)
                model_file = f'models_{VERSION_NAME}/dqn_model_{episode}.pt'
                agent.save(model_file)
                plt.close()  # to avoid memory errors
        
        # Reset the environment
        obv, _ = env.reset()
        total_reward = 0
        if not learning:
            env.pyrace.mode = 2  # continuous display of game
        
        for t in range(max_t):
            # Select and perform an action
            action = agent.select_action(obv, explore=learning)
            next_obv, reward, done, _, info = env.step(action)
            env.remember(obv, action, reward, next_obv, done)
            
            if learning:
                # Store the transition in memory
                agent.remember(obv, action, reward, next_obv, done)
                
                # Perform one step of the optimization
                agent.replay()
                
            # Move to the next state
            obv = next_obv
            total_reward += reward
            
            # Display the game
            if (episode % DISPLAY_EPISODES == 0) or (env.pyrace.mode == 2):
                env.set_msgs(['SIMULATE',
                             f'Episode: {episode}',
                             f'Time steps: {t}',
                             f'check: {info["check"]}',
                             f'dist: {info["dist"]}',
                             f'crash: {info["crash"]}',
                             f'Reward: {total_reward:.0f}',
                             f'Max Reward: {max_reward:.0f}',
                             f'Epsilon: {agent.epsilon:.4f}'])
                env.render()
            
            if done or t >= max_t - 1:
                if total_reward > max_reward:
                    max_reward = total_reward
                break


def load_and_play(episode, learning=False):
    # Load a saved model and play the game
    agent = DQNAgent(state_dim=len(env.observation_space.low), action_dim=env.action_space.n)
    model_file = f'models_{VERSION_NAME}/dqn_model_{episode}.pt'
    agent.load(model_file)
    
    # Play the game
    simulate(agent, env, learning=learning, episode_start=episode)


if __name__ == "__main__":
    # Initialize the environment
    env = gym.make("Pyrace-v1").unwrapped  # skip the TimeLimit and OrderEnforcing default wrappers
    print('env', type(env))
    
    # Create directories for saving models
    if not os.path.exists(f'models_{VERSION_NAME}'):
        os.makedirs(f'models_{VERSION_NAME}')
    
    # Print environment information
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    
    # Hyperparameters
    NUM_EPISODES = 65_000
    MAX_T = 2000
    GAMMA = 0.99            # discount factor
    EPSILON_START = 1.0     # start with 100% exploration
    EPSILON_MIN = 0.01      # minimum exploration rate
    EPSILON_DECAY = 0.995   # decay rate for exploration
    LEARNING_RATE = 0.001   # learning rate for optimizer
    BUFFER_SIZE = 100000    # replay buffer size
    BATCH_SIZE = 64         # minibatch size for training
    
    # Initialize the DQN agent
    agent = DQNAgent(
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
    
    # Choose what to do:
    
    # Train a new model
    simulate(agent, env, learning=True, num_episodes=NUM_EPISODES, max_t=MAX_T)
    
    # Alternatively, load and continue training a saved model
    # load_and_play(3500, learning=True)
    
    # Or just play with a saved model without learning
    # load_and_play(3500, learning=False) 