import random
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

from config import GRID_SIZE, reward_for_eating, reward_for_dying


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
        self._model.fit(np.array([0] * 24).reshape(1, self.input_neurons),
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


        food_right, border_right, self_right = Agent.look_direction((GRID_SIZE, 0), food, snake, height, width)
        food_up, border_up, self_up = Agent.look_direction((0, -GRID_SIZE), food, snake, height, width)
        food_left, border_left, self_left = Agent.look_direction((-GRID_SIZE, 0), food, snake, height, width)
        food_down, border_down, self_down = Agent.look_direction((0, GRID_SIZE), food, snake, height, width)
        food_right_up, border_right_up, self_right_up = Agent.look_direction((GRID_SIZE, -GRID_SIZE), food, snake, height, width)
        food_left_up, border_left_up, self_left_up = Agent.look_direction((-GRID_SIZE, -GRID_SIZE), food, snake, height, width)
        food_right_down, border_right_down, self_right_down = Agent.look_direction((GRID_SIZE, GRID_SIZE), food, snake, height, width)
        food_left_down, border_left_down, self_left_down = Agent.look_direction((-GRID_SIZE, GRID_SIZE), food, snake, height, width)

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

        return np.array(state)


    @staticmethod
    def in_bounds(x, y, width, height):
        return 0 <= x < width and 0 <= y < height


    @staticmethod
    def look_direction(direction, food, snake, height, width):
        food_x = food.position()[0]
        food_y = food.position()[1]

        cur_x, cur_y = snake.head_pos()[0], snake.head_pos()[1]
        cur_x += direction[0]
        cur_y += direction[1]
        distance = 1
        food_dir = snake_dir = 0
        while Agent.in_bounds(cur_x, cur_y, width, height):

            if (food_dir == 0) and (cur_x == food_x and cur_y == food_y):
                food_dir = 1

            if (snake_dir == 0) and ((cur_x, cur_y) in snake.positions()[1:]):
                snake_dir = 1/distance

            cur_x += direction[0]
            cur_y += direction[1]
            distance += 1

        wall_dir = 1/distance

        return food_dir, wall_dir, snake_dir



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

        action_value = (c_prev - c_next)/10
        if action_value > 0:
            action_value = 0
        reward += action_value
        return reward

    @staticmethod
    def rew(food, prev_pos, snake, prev_state):

        reward = Agent.initial_reward(snake)

        x_food = food.position()[0]
        y_food = food.position()[1]

        x_prev = prev_pos[0]
        y_prev = prev_pos[1]

        x_next = snake.head_pos()[0]
        y_next = snake.head_pos()[1]

        diff_x = np.abs(x_food - x_prev)
        diff_y = np.abs(y_food - y_prev)

        if 1 in prev_state[0:8]:
            next_diff_x = np.abs(x_food - x_next)
            next_diff_y = np.abs(y_food - y_next)
            if next_diff_x < diff_x or next_diff_y < diff_y:
                reward += 5
            else:
                reward -= 5
        return reward

    @staticmethod
    def initial_reward(snake):
        reward = 0
        if snake.eaten():
            reward = reward_for_eating
        if snake.done():
            reward = reward_for_dying
        return reward


    def plot(self):
        losses = pd.DataFrame(self._model.history.history)
        losses.plot()
        plt.show()


    def memory(self):
        return self._memory

    def get_model(self):
        return self._model
