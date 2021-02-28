import random
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential


class Agent(object):

    def __init__(self,
                 weights_path='./weights/weights.h5',
                 load=True,
                 learning_rate=0.0005,
                 gamma=0.9,
                 neurons_each_layer=(12, 50, 300, 50),
                 batch_size=1000):

        self.input_neurons = neurons_each_layer[0]
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._memory = collections.deque(maxlen=2500)
        self._batch_size = batch_size
        self._weights_path = weights_path

        self._model = self.model(Adam(learning_rate), neurons_each_layer)
        if load:
            self.load_model()


    def model(self, optimizer, neurons_each_layer):
        model = Sequential()
        for no_of_neurons in neurons_each_layer:
            model.add(Dense(no_of_neurons, activation='relu'))
        model.add(Dense(4, activation='softmax'))
        model.compile(loss='mse', optimizer=optimizer)

        return model


    def save_model(self):
        self._model.save_weights(self._weights_path)


    def load_model(self):
        self._model.fit(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]).reshape(1, self.input_neurons),
                        np.array([0., 0., 1., 0.]).reshape(1, 4), epochs=1, verbose=0)
        self._model.load_weights(self._weights_path)


    def remember(self, decision):
        self._memory.append(decision)


    def train(self, prev_state, action, reward, new_state, done):
        target = reward
        if not done:
            target = reward + self._gamma * \
                     np.amax(self._model.predict(np.array(new_state).reshape((1, self.input_neurons)))[0])
        target_f = self._model.predict(np.array(prev_state).reshape((1, self.input_neurons)))
        target_f[0][np.argmax(action)] = target
        self._model.fit(prev_state.reshape((1, self.input_neurons)), target_f, epochs=1, verbose=0)


    def long_train(self, memory):
        if len(self._memory) > self._batch_size:
            minibatches = random.sample(memory, self._batch_size)
        else:
            minibatches = memory
        for prev_state, action, reward, new_state, done in minibatches:
            self.train(prev_state, action, reward, new_state, done)


    @staticmethod
    def state(width, height, snake, food):

        x, y = tuple(snake.head_pos())

        danger_left = (x <= 0) or \
                      ((x - 20, y) in snake.positions())

        danger_front = (y <= 0) or \
                       ((x, y - 20) in snake.positions())

        danger_right = (x >= width - 20) or \
                       ((x + 20, y) in snake.positions())

        danger_down = (y >= height - 20) or \
                      ((x, y + 20) in snake.positions())

        food_left = food.position()[0] < x

        food_right = food.position()[0] > x

        food_down = food.position()[1] > y

        food_front = food.position()[1] < y

        state = [
            danger_left,
            danger_right,
            danger_front,
            danger_down,

            snake.moving_left,
            snake.moving_right,
            snake.moving_down,
            snake.moving_up,

            food_left,
            food_right,
            food_down,
            food_front
        ]

        state = Agent.state_to_int(state)

        return state


    @staticmethod
    def state_to_int(state):
        for i in state:
            if state[i]:
                state[i] = 1
            else:
                state[i] = 0
        return np.asarray(state)


    @staticmethod
    def reward(snake, food, height, prev_pos):
        reward = Agent.initial_reward(snake)

        """Checking if snake came closer to food"""
        x_food = food.position()[0]
        y_food = height - food.position()[1]

        x_prev = prev_pos[0]
        y_prev = height - prev_pos[1]

        x_next = snake.head_pos()[0]
        y_next = height - snake.head_pos()[1]

        x = np.abs(x_prev - x_food)
        y = np.abs(y_prev - y_food)

        c_prev = np.sqrt(np.square(x) + np.square(y))

        x = np.abs(x_next - x_food)
        y = np.abs(y_next - y_food)

        c_next = np.sqrt(np.square(x) + np.square(y))

        action_value = c_prev - c_next
        if action_value < 0:
            action_value = -0.1

        reward += action_value
        return reward


    @staticmethod
    def initial_reward(snake):
        reward = 0
        if snake.eaten():
            reward = 50
        if snake.done():
            reward = -100
        return reward


    def plot(self):
        losses = pd.DataFrame(self._model.history.history)
        losses.plot()
        plt.show()


    def memory(self):
        return self._memory


    def get_model(self):
        return self._model
