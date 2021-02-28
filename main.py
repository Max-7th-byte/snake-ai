import random
from copy import deepcopy
import threading
import os

import numpy as np
from tensorflow.keras.utils import to_categorical

from models.ai import Agent
from views.display import *
import config
from models.highlights import Highlight, add, get_highlights, remember_highlights
from models.telegram.telegram import report
from views.game import Snake, Food, SCREEN_WIDTH, SCREEN_HEIGHT, map_direction


def main():
    if not replay_best_plays:
        current_generation = 0
        snake = None
        agent = None
        scores = []
        highlight = Highlight()

        while current_generation < NO_OF_GEN:
            agents = []
            snakes = []
            no_snake = 0
            while no_snake < POPULATION:
                weights_path = f'/home/max/IdeaProjects/snake_evolution_refactored_1/models/weights/tmp/weights{no_snake}.h5'
                if display:
                    agent, _snake = display_train(highlight, agent, current_generation, no_snake, weights_path)
                else:
                    agent, _snake = non_display_train(highlight, agent, current_generation, no_snake, weights_path)
                agents.append(agent)
                snakes.append(_snake)
                no_snake += 1
            ind, snake, agent = pick_best(snakes, agents)
            config.load = True

            stream = open(f'/home/max/IdeaProjects/snake_evolution_refactored_1/models/weights/tmp/weights{ind}.h5', 'rb')
            for i, file_path in enumerate(os.listdir('/home/max/IdeaProjects/snake_evolution_refactored_1/models/weights/tmp')):
                with open('/home/max/IdeaProjects/snake_evolution_refactored_1/models/weights/tmp/' + file_path, 'wb') as f:
                    f.write(stream.read())
            stream.close()
            current_generation += 1
            scores.append([snake.score() for snake in snakes])
        final_score(snake, scores, current_generation * POPULATION)
        remember_highlights()
    else:
        highlights = get_highlights()
        for h in highlights:
            print('H')
            h.replay()


def train(highlight, snake, current_generation, no_snake, agents, snakes, weights_path):
    if display:
        agent, _snake = display_train(highlight, snake, current_generation, no_snake, weights_path)
    else:
        agent, _snake = non_display_train(highlight, snake, current_generation, no_snake, weights_path)
    agents.append(agent)
    snakes.append(_snake)
    no_snake += 1


def display_train(highlight, agent=None, current_generation=0, no_snake=0,
                  weights_path='/home/max/IdeaProjects/snake_evolution_refactored_1/models/weights/weights0.h5'):
    clock, screen, font = initialize_display(SCREEN_WIDTH, SCREEN_HEIGHT)

    if agent is None:
        agent = Agent(weights_path, load, learning_rate, gamma, neurons_each_layer, batch_size)

    surface, game = draw_surface(screen, False)

    snake = Snake(snake_color, game)
    food = Food(food_color, snake)
    highlight.remember(deepcopy(food))

    while not snake.done():
        clock.tick(speed)

        if is_human_playing:
            snake.handle_keys(food)
        else:
            perform_action(agent, snake, food, highlight)
        update_display(snake, food, surface, screen, font)

    score = update_parameters(agent, snake, highlight)
    print_result(current_generation, no_snake, score)

    snake.reset()
    pygame.display.update()
    save(agent, weights_path)

    return agent, snake


def non_display_train(highlight, agent=None, current_generation=0, no_snake=0,
                      weights_path='/home/max/IdeaProjects/snake_evolution_refactored_1/models/weights/weights0.h5'):

    if agent is None:
        agent = Agent(weights_path, load, learning_rate, gamma, neurons_each_layer, batch_size)

    snake = Snake(snake_color, game=None)
    food = Food(food_color, snake)
    highlight.remember(deepcopy(food))

    while not snake.done():
        perform_action(agent, snake, food, highlight)

    score = update_parameters(agent, snake, highlight)
    print_result(current_generation, no_snake, score)

    save(agent, weights_path)
    snake.reset()

    return agent, snake


def pick_best(snakes, agents):
    snake = None
    agent = None
    record = 0
    ind = 0
    i = 0
    while i < len(snakes):
        if snakes[i].score() >= record:
            snake = snakes[i]
            record = snakes[i].score()
            ind = i
            agent = agents[i]
        i += 1
    return ind, snake, agent


def perform_action(agent, snake, food, highlight):

    prev_state = agent.state(SCREEN_WIDTH, SCREEN_HEIGHT, snake, food)
    prev_pos = snake.head_pos()
    action = pick_action(prev_state, agent)
    direct, f = direction(action), deepcopy(food)
    snake.turn(direction(action), food)
    highlight.remember((direct, f, food.position()))

    if training:
        short_train(agent, snake, food, action, prev_pos, prev_state)


def pick_epsilon(played_games, no_of_games):
    if not training:
        epsilon = 0.00
    else:
        epsilon = 0.8 - played_games * 1 / (no_of_games / 3)
    return epsilon


def pick_action(prev_state, agent):

    prediction = agent.get_model().predict(prev_state.reshape(1, agent.input_neurons))
    action = to_categorical(np.argmax(prediction[0]), num_classes=4)
    return action


def direction(action):
    return map_direction.get(np.argmax(action))


def short_train(agent, snake, food, action, prev_pos, prev_state):
    new_state = agent.state(SCREEN_WIDTH, SCREEN_HEIGHT, snake, food)
    reward = agent.reward(snake, food, SCREEN_HEIGHT, prev_pos)
    agent.train(prev_state, action, reward, new_state, snake.done)
    agent.remember((prev_state, action, reward, new_state, snake.done))


def print_result(current_generation, no_snake, score):
    print(f'Generation: {current_generation}\t\tSnake: {no_snake}\t\tScore {score}')


def prepare_info(scores, record, played_games):
    text_report = ''
    for game, score in enumerate(scores[played_games - save_period:]):
        text_report += f'Game: {played_games - save_period + game + 1}\t\tScore: {score}\n'
    text_report += f'\n\nHighest score: {record}' \
                   f'\nMean: {np.mean(scores[played_games - save_period:])}' \
                   f'\nStd: {np.std(scores[played_games - save_period:])}'
    return text_report


def update_parameters(agent, snake, highlight):
    if training:
        agent.long_train(agent.memory())
    score = snake.length() - 1
    if score >= highlight_score_lower_limit:
        add(highlight)



    return score


def save(agent, weights_path):
    agent.get_model().save_weights(weights_path)


def final_score(snake, scores, played_games):
    print(f'Highest score: {snake.score()}\t\t\t', end='')
    print(f'Mean: {np.mean(np.array(scores))}\t\tStd: {np.std(np.array(scores))}')
    report(f'Finished training for {played_games} games\n'
           f'Highest score: {snake.score()}\n'
           f'Mean: {np.mean(np.array(scores))}\nStd: {np.std(np.array(scores))}')


if __name__ == '__main__':
    main()
