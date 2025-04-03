import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_race.envs.pyrace_2d_continuous import PyRace2DContinuous

class RaceEnvContinuous(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}
    
    def __init__(self, render_mode="human", ):
        print("Initializing RaceEnvContinuous")
        # Expanded action space to include braking: 
        # 0: accelerate, 1: turn left, 2: turn right, 3: brake
        self.action_space = spaces.Discrete(4)
        
        # Continuous observation space with radar distances (5 directions) + speed + steering angle
        # Each radar gives raw distance values in pixels (0-200)
        # Also includes current speed (0-10) and current steering angle
        self.observation_space = spaces.Box(
            np.array([0, 0, 0, 0, 0, 0, 0]), 
            np.array([200, 200, 200, 200, 200, 10, 360]), 
            dtype=np.float32
        )
        
        self.is_view = True
        self.pyrace = PyRace2DContinuous(self.is_view)
        self.memory = []
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        mode = self.pyrace.mode
        del self.pyrace
        self.is_view = True
        self.msgs = []
        self.pyrace = PyRace2DContinuous(self.is_view, mode=self.render_mode)
        obs = self.pyrace.observe()
        return obs, {}

    def step(self, action):
        self.pyrace.action(action)
        reward = self.pyrace.evaluate()
        done = self.pyrace.is_done()
        obs = self.pyrace.observe()
        
        # Return more detailed info dictionary
        info = {
            'dist': self.pyrace.car.distance,
            'check': self.pyrace.car.current_check,
            'crash': not self.pyrace.car.is_alive,
            'speed': self.pyrace.car.speed,
            'angle': self.pyrace.car.angle,
            'time': self.pyrace.car.time_spent
        }
        
        return obs, reward, done, False, info

    def render(self):
        if self.is_view:
            self.pyrace.view_(self.msgs)

    def set_view(self, flag):
        self.is_view = flag

    def set_msgs(self, msgs):
        self.msgs = msgs

    def save_memory(self, file):
        np.save(file, np.array(self.memory, dtype=object))
        print(file + " saved")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) 