import sys, os
import numpy as np
import torch
import gymnasium as gym
import gym_race
import time
import argparse

# Import the required classes from the training script - remove check_point
from Pyrace_RL_DQN import ImprovedDQNAgent, ImprovedDQNWithLayerNorm

# Default configuration
MODEL_VERSION = 'DQN_v04b'
MODEL_EPISODE = 1100
USE_LAYER_NORM = True
DISPLAY_SLEEP = 0.05  # Set to 0 for fastest gameplay or higher (e.g., 0.1) for slower viewing

def run_model(model_version=MODEL_VERSION, model_episode=MODEL_EPISODE, 
              num_episodes=1, display_sleep=DISPLAY_SLEEP):
    print(f"Running model: models_{model_version}/dqn_model_{model_episode}.pt")
    print(f"Episodes to run: {num_episodes}")
    
    # Initialize environment
    env = gym.make("Pyrace-v3").unwrapped
    
    # Import check_point from the environment
    from gym_race.envs.pyrace_2d_continuous import check_point
    
    env.pyrace.mode = 2  # Enable display
    
    # Create agent with the same architecture as during training
    agent = ImprovedDQNAgent(
        state_dim=len(env.observation_space.low),
        action_dim=env.action_space.n,
        use_layer_norm=USE_LAYER_NORM
    )
    
    # Load the model
    model_file = f'models_{model_version}/dqn_model_{model_episode}.pt'
    try:
        agent.load(model_file)
        print(f"Successfully loaded model from {model_file}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set epsilon to minimum for exploitation only (no exploration)
    agent.steps_done = agent.epsilon_decay
    
    # Statistics across multiple episodes
    success_count = 0
    crash_count = 0
    timeout_count = 0
    all_rewards = []
    all_steps = []
    all_checkpoints = []
    
    # Run episodes
    for episode in range(num_episodes):
        print(f"\nRunning episode {episode+1}/{num_episodes}...")
        
        obv, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        max_steps = 2000  # Same as during training
        
        while not done and step_count < max_steps:
            # Select action (no exploration)
            action = agent.select_action(obv, explore=False)
            
            # Take action
            next_obv, reward, done, _, info = env.step(action)
            
            # Update stats
            total_reward += reward
            step_count += 1
            
            # Display information
            msgs = [
                f'BEST MODEL: {model_version} Episode {model_episode}',
                f'Run: {episode+1}/{num_episodes}',
                f'Time steps: {step_count}',
                f'Check: {info["check"]}/{len(check_point)}',
                f'Dist: {info["dist"]:.1f}',
                f'Speed: {info["speed"]:.1f}',
                f'Crash: {info["crash"]}',
                f'Reward: {total_reward:.0f}'
            ]
            
            env.set_msgs(msgs)
            env.render()
            
            # Add a sleep to make it easier to watch
            time.sleep(display_sleep)
            
            # Update observation
            obv = next_obv
        
        # Track episode outcomes
        if done and info["check"] >= len(check_point):
            print(f"SUCCESS! Completed all {len(check_point)} checkpoints in {step_count} steps!")
            success_count += 1
        elif done and info["crash"]:
            print(f"CRASH at checkpoint {info['check']}/{len(check_point)}")
            crash_count += 1
        else:
            print(f"Reached max steps ({max_steps}) with {info['check']}/{len(check_point)} checkpoints")
            timeout_count += 1
        
        # Record episode stats
        all_rewards.append(total_reward)
        all_steps.append(step_count)
        all_checkpoints.append(info["check"])
        
        print(f"Episode {episode+1} reward: {total_reward:.0f}")
    
    # Print overall statistics
    print("\n===== Overall Statistics =====")
    print(f"Episodes run: {num_episodes}")
    print(f"Success rate: {success_count}/{num_episodes} ({success_count/num_episodes:.0%})")
    print(f"Crash rate: {crash_count}/{num_episodes} ({crash_count/num_episodes:.0%})")
    print(f"Timeout rate: {timeout_count}/{num_episodes} ({timeout_count/num_episodes:.0%})")
    print(f"Average reward: {np.mean(all_rewards):.0f}")
    print(f"Average steps: {np.mean(all_steps):.0f}")
    print(f"Average checkpoints: {np.mean(all_checkpoints):.1f}/{len(check_point)}")
    
    print("\nPress any key to exit...")
    # Keep the window open until user presses a key
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                waiting = False
                break

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the best trained model for PyRace')
    parser.add_argument('--version', type=str, default=MODEL_VERSION, 
                        help=f'Model version name (default: {MODEL_VERSION})')
    parser.add_argument('--episode', type=int, default=MODEL_EPISODE, 
                        help=f'Episode number to load (default: {MODEL_EPISODE})')
    parser.add_argument('--runs', type=int, default=1, 
                        help='Number of episodes to run (default: 1)')
    parser.add_argument('--sleep', type=float, default=DISPLAY_SLEEP, 
                        help=f'Sleep time between frames (default: {DISPLAY_SLEEP})')
    args = parser.parse_args()
    
    # Import pygame here to avoid potential import issues
    import pygame
    
    try:
        # Run the model with the specified parameters
        run_model(
            model_version=args.version, 
            model_episode=args.episode,
            num_episodes=args.runs,
            display_sleep=args.sleep
        )
    except KeyboardInterrupt:
        print("\nRun interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc() 