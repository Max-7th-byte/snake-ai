import random
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

from config import reward_for_away_from_food, reward_for_closer_to_food, GRID_SIZE


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
        self._model.fit(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0]).reshape(1, self.input_neurons),
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

        food_x = food.position()[0]
        food_y = food.position()[1]

        snake_x = snake.head_pos()[0]
        snake_y = height - snake.head_pos()[1]

        food_right = food_x - snake_x if food_x - snake_x > 0 and food_y != snake_y else 0
        food_up = snake_y - food_y if snake_y - food_y > 0 and food_x != snake_x else 0
        food_left = snake_x - food_x if snake_x - food_x > 0 and food_y != snake_y else 0
        food_down = food_y - snake_y if food_y - snake_y > 0 and food_x != snake_x else 0

        border_right = width - snake_x
        border_up = snake_y
        border_left = snake_x
        border_down = height - snake_y

        self_right = 0
        self_up = 0
        self_left = 0
        self_down = 0
        x_right = x_left = x_up = x_down = snake_x
        y_right = y_left = y_up = y_down = snake_y
        while Agent.in_bounds(x_right, x_left, x_up, x_down, y_right, y_left, y_up, y_down, width, height):

            if Agent.in_bounds(x_right, y_right, width, height):
                if self_right == 0 and (x_right, y_right) in snake.positions()[1:]:
                    self_right = x_right - snake_x
                x_right += GRID_SIZE

            if Agent.in_bounds(x_left, y_left, width, height):
                if self_left == 0 and (x_left, y_left) in snake.positions()[1:]:
                    self_left = snake_x - x_left
                x_left -= GRID_SIZE

            if Agent.in_bounds(x_up, y_up, width, height):
                if self_up == 0 and (x_up, y_up) in snake.positions()[1:]:
                    self_up = snake_y - y_up
                y_up -= GRID_SIZE

            if Agent.in_bounds(x_down, y_down, width, height):
                if self_down == 0 and (x_down, y_down) in snake.positions()[1:]:
                    self_down = y_down - snake_y
                y_down += GRID_SIZE


        food_right_up = 0
        border_right_up = 0
        self_right_up = 0


        food_left_up = 0
        border_left_up = 0
        self_left_up = 0

        food_right_down = 0
        border_right_down = 0
        self_right_down = 0

        food_left_down = 0
        border_left_down = 0
        self_left_down = 0

        x_right_up = x_left_up = x_right_down = x_left_down = snake_x
        y_right_up = y_left_up = y_right_down = y_left_down = snake_y
        while Agent.in_bounds(x_right_up, x_left_up, x_right_down, x_left_down, y_right_up, y_left_up, y_right_down, y_left_down, width, height):

            if Agent.in_bounds(x_right_up, y_right_up, width, height):
                if food_right_up == 0 and (x_right_up, y_right_up) == food.position():
                    pass

            if Agent.in_bounds(x_left_up, y_left_up, width, height):
                pass

            if Agent.in_bounds(x_right_down, y_right_down, width, height):
                pass

            if Agent.in_bounds(x_left_down, y_left_down, width, height):
                pass

        state = [
            food_right,
            food_up,
            food_left,
            food_down,
            food_right_up,
            food_left_up,
            food_right_down,
            food_left_down,
            border_right,
            border_up,
            border_left,
            border_down,
            border_right_up,
            border_left_up,
            border_right_down,
            border_left_down,
            self_right,
            self_up,
            self_left,
            self_down,
            self_right_up,
            self_left_up,
            self_right_down,
            self_left_down
        ]

        return state


    @staticmethod
    def in_bounds(x_right, x_left, x_up, x_down, y_right, y_left, y_up, y_down, width, height):
        return Agent.in_bounds(x_right, y_right, width, height) or Agent.in_bounds(x_left, y_left, width, height) or \
               Agent.in_bounds(x_up, y_up, width, height) or Agent.in_bounds(x_down, y_down, width, height)

    @staticmethod
    def in_bounds(x, y, width, height):
        return 0 <= x <= width or 0 <= y <= height

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
        if action_value > 0:
            action_value = reward_for_closer_to_food
        else:
            action_value = reward_for_away_from_food

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
