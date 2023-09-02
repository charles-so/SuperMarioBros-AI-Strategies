# Q - learning algorithm for super mario bros

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import numpy as np


env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

print("Action space: ", SIMPLE_MOVEMENT)
print("Observation Space", env.observation_space.n)
print("Sample observation", env.observation_space.sample())



# whether to reset environment or not
done = True
env.reset()
for step in range(5000):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    # state is the new frame
    # reward is the reward for the action
    # terminated is whether the episode is over (we're dead or game is done)
    done = terminated or truncated
    if done:
        state = env.reset()
env.close()