# Wrapper functions for Q-learning and super mario gym
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

import numpy as np
import tensorflow as tf
from gym.spaces import Box
import matplotlib.pyplot as plt

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, truncated, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = tf.convert_to_tensor(observation, dtype=tf.float32)
        return observation

    def observation(self, observation):
        #print(f"Original observation shape: {observation.shape}")  # Debug line
        observation = tf.convert_to_tensor(observation, dtype=tf.float32)
        #print(f"Shape after permute_orientation: {observation.shape}")  # Debug line
        observation = tf.image.rgb_to_grayscale(observation)
        # plt.imshow(observation, cmap='gray')
        # plt.title("Original Frame")
        # plt.axis('off')
        # plt.show()
        # plt.pause(5)

        return observation
    

class DownSampleObservation(gym.ObservationWrapper):
    def __init__(self, env, downsample_rate):
        super().__init__(env)
        self.downsample_rate = downsample_rate
        obs_shape = self.observation_space.shape
        # New shape after downsampling
        new_shape = (obs_shape[0] // downsample_rate, obs_shape[1] // downsample_rate, obs_shape[2])
        self.observation_space = Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        # Downsample the observation by taking every nth pixel in each dimension
        observation = observation[::self.downsample_rate, ::self.downsample_rate, :]
        return observation
    



# Assume `env` is your original environment
# Apply Wrappers to environment

# if gym.__version__ < '0.26':
#     env = FrameStack(env, num_stack=4, new_step_api=True)
# else:
    
