import time
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class DQNAgent:
    def __init__(self, states, actions, max_memory, double_q):
        self.states = states
        self.actions = actions
        self.memory = deque(maxlen=max_memory)
        self.eps = 1.0
        self.eps_decay = 0.99999975
        self.eps_min = 0.1
        self.gamma = 0.90
        self.batch_size = 32
        self.burnin = 100000
        self.copy = 10000
        self.step = 0
        self.learn_each = 3
        self.learn_step = 0
        self.save_each = 500000
        self.double_q = double_q
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.sync_target()

    def build_model(self):
        input_data = layers.Input(shape=self.states)
        x = input_data / 255.0  # Normalize
        x = layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(x)
        x = layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        output = layers.Dense(self.actions, activation='linear')(x)
        model = Model(input_data, output)
        model.compile(optimizer=Adam(learning_rate=0.00025), loss='huber_loss')
        return model

    def sync_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def add(self, experience):
        self.memory.append(experience)

    def predict(self, model, state):
        return model.predict(np.array(state))

    def run(self, state):
        if np.random.rand() < self.eps:
            action = np.random.randint(low=0, high=self.actions)
        else:
            q_values = self.predict(self.model, np.expand_dims(state, 0))
            action = np.argmax(q_values)

        self.eps *= self.eps_decay
        self.eps = max(self.eps_min, self.eps)
        self.step += 1
        return action

    def learn(self):
        if self.step % self.copy == 0:
            self.sync_target()
        if self.step % self.save_each == 0:
            self.model.save(f'./models/model_step_{self.step}.h5')
        if self.step < self.burnin:
            return
        if self.learn_step < self.learn_each:
            self.learn_step += 1
            return

        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(np.array, zip(*batch))
        next_q_values = self.predict(self.target_model, next_state)

        if self.double_q:
            online_q_values = self.predict(self.model, next_state)
            a = np.argmax(online_q_values, axis=1)
            target_q = reward + (1. - done) * self.gamma * next_q_values[np.arange(0, self.batch_size), a]
        else:
            target_q = reward + (1. - done) * self.gamma * np.amax(next_q_values, axis=1)

        target_f = self.predict(self.model, state)
        for i, val in enumerate(action):
            target_f[i][val] = target_q[i]

        self.model.train_on_batch(state, target_f)
        self.learn_step = 0
