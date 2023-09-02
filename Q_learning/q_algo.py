# Q - learning algorithm for super mario bros
import gym
from gym.wrappers import FrameStack
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# to make plotlib work
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from q_wrapper import SkipFrame, GrayScaleObservation, DownSampleObservation


env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Call Wrappers
env = SkipFrame(env, skip=4)
env = DownSampleObservation(env, downsample_rate=2) #(240, 256, 3)  2-> (120, 128, 3)
env = GrayScaleObservation(env)
env = FrameStack(env, num_stack=4)

# whether to reset environment or not
done = True
env.reset()
for step in range(10):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    print(state.shape)

    # show the first frame in the stack
    # plt.imshow(state[0], cmap='gray')  
    # plt.show()
    # state is the new frame
    # reward is the reward for the action
    # terminated is whether the episode is over (we're dead or game is done)
    input("Press Enter to continue...")
    done = terminated or truncated
    if done:
        state = env.reset()
        break


env.close()