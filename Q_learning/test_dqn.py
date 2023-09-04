# Q - learning algorithm for super mario bros
import gym
from gym.wrappers import FrameStack
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

from q_wrapper import SkipFrame, GrayScaleObservation, DownSampleObservation

# load pretrained model
save_path = "saved_models"
model = tf.keras.models.load_model(os.path.join(save_path, "final_model.h5"))

# make environment
env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Call Wrappers
env = SkipFrame(env, skip=4)
env = DownSampleObservation(env, downsample_rate=2) #(240, 256, 3)  2-> (120, 128, 3)
env = GrayScaleObservation(env)
env = FrameStack(env, num_stack=4)

state, _ = env.reset()
state = np.array(state)
state = np.squeeze(state, axis=-1)
state = np.transpose(state, (1, 2, 0))
print("Observation space:", env.observation_space)
done = False
total_reward = 0

while not done:
    env.render()
    state_for_prediction = np.expand_dims(state, axis=0)  # Add a batch dimension

    # Choose the action based on Q-values (No exploration)
    q_values = model.predict(state_for_prediction)
    action = np.argmax(q_values[0])

    # Take action, observe new state and reward
    next_state, reward, done, truncated, info = env.step(action)
    next_state = np.array(next_state)
    next_state = np.squeeze(next_state, axis=-1)
    next_state = np.transpose(next_state, (1, 2, 0))

    total_reward += reward

    state = next_state

print(f"Episode: {episode+1}, Total Reward: {total_reward}")
env.close()