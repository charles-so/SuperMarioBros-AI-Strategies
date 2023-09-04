# Q - learning algorithm for super mario bros
import gym
from gym.wrappers import FrameStack
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from nes_py.wrappers import JoypadSpace

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# to make plotlib work
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from q_wrapper import SkipFrame, GrayScaleObservation, DownSampleObservation
from agent_dqn_TF2 import DQNAgent

# Save path for models after training
save_path = "saved_models"
if not os.path.exists(save_path):
    os.makedirs(save_path)

env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Call Wrappers
env = SkipFrame(env, skip=4)
env = DownSampleObservation(env, downsample_rate=2) #(240, 256, 3)  2-> (120, 128, 3)
env = GrayScaleObservation(env)
env = FrameStack(env, num_stack=4)

print("Observation space:", env.observation_space)

# Parameters
states = (120, 128, 4)
actions = env.action_space.n

# Agent
agent = DQNAgent(states=states, actions=actions, max_memory=100000, double_q=False)

# Episodes
episodes = 10000
rewards = []

# Timing
start = time.time()
step = 0

for e in range(episodes):

    # Reset env
    # done = True
    state, _ = env.reset()
    state = np.array(state)
    state = np.squeeze(state, axis=-1)
    state = np.transpose(state, (1, 2, 0))

    # Reward
    total_reward = 0
    iter = 0

    done = True

    while True:
        # Show env (disabled)
        # env.render()

        # Run agent
        action = agent.run(state=state)
        #print(state.shape)
        #print(type(state))
        #print(state)
        next_state, reward, terminated, truncated, info = env.step(action = action)
        next_state = np.array(next_state)
        next_state = np.squeeze(next_state, axis=-1)
        next_state = np.transpose(next_state, (1, 2, 0))

        # Remember transition
        agent.add(experience=(state, next_state, action, reward, terminated))

        # Update agent
        agent.learn()

        # Total reward
        total_reward += reward

        # Update state
        state = next_state

        # Increment
        iter += 1

        # If done break loop
        if done or info['flag_get']:
            break
    
    # Rewards
    rewards.append(total_reward / iter)

    # Print
    if e % 100 == 0:
        print('Episode {e} - +'
              'Frame {f} - +'
              'Frames/sec {fs} - +'
              'Epsilon {eps} - +'
              'Mean Reward {r}'.format(e=e,
                                       f=agent.step,
                                       fs=np.round((agent.step - step) / (time.time() - start)),
                                       eps=np.round(agent.eps, 4),
                                       r=np.mean(rewards[-100:])))
        start = time.time()
        step = agent.step
    
agent.model.save(os.path.join(save_path, "final_model.h5"))
np.save('rewards.npy', rewards)
env.close()
