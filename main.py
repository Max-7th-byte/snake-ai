import random
from copy import deepcopy
import time

import numpy as np
from tensorflow.keras.utils import to_categorical

from models.ai import Agent
from views.display import *
from models.highlights import Highlight, add, get_highlights, remember_highlights
from models.telegram.telegram import report
from views.game import Snake, Food, SCREEN_WIDTH, SCREEN_HEIGHT, map_direction


def main():
    if not replay_best_plays:
        if display:
            display_main()
        else:
            non_display_main()
    else:
        highlights = get_highlights()
        for h in highlights:
            h.replay()


def display_main():
    clock, screen, font = initialize_display(SCREEN_WIDTH, SCREEN_HEIGHT)

    agent = Agent(weights_path, load, learning_rate, gamma, neurons_each_layer, batch_size)
    played_games = 0
    scores = list()
    record = 0

    while played_games < number_of_games:
        surface, game = draw_surface(screen, False)
        highlight = Highlight()

        snake = Snake(snake_color, game)
        food = Food(food_color, snake)
        highlight.remember(deepcopy(food))

        initial_time = time.time()
        while not snake.done():
            clock.tick(speed)

            if is_human_playing:
                agent.state(SCREEN_WIDTH, SCREEN_HEIGHT, snake, food)
                snake.handle_keys(food)
            else:
                current_time = time.time()
                if current_time - initial_time > 500:
                    break
                perform_action(agent, snake, food, played_games, highlight)
            update_display(snake, food, surface, screen, font)

        played_games, score, record = update_parameters(agent, played_games, snake,
                                                        scores, record, highlight)
        print_result(played_games, score)

        snake.reset()
        pygame.display.update()
        save(agent, played_games, scores, record)

    save(agent, played_games, scores, record)
    final_score(scores, record, played_games)



def non_display_main():
    agent = Agent(weights_path, load, learning_rate, gamma, neurons_each_layer, batch_size)
    played_games = 0
    scores = list()
    record = 0

    while played_games < number_of_games:

        highlight = Highlight()
        snake = Snake(snake_color, game=None)
        food = Food(food_color, snake)
        highlight.remember(deepcopy(food))

        initial_time = time.time()
        while not snake.done():
            current_time = time.time()
            if current_time - initial_time > 300:
                break
            perform_action(agent, snake, food, played_games, highlight)

        played_games, score, record = update_parameters(agent, played_games, snake,
                                                        scores, record, highlight)
        print_result(played_games, score)

        snake.reset()
        save(agent, played_games, scores, record)

    save(agent, played_games, scores, record)
    final_score(scores, record, played_games)


def perform_action(agent, snake, food, played_games, highlight):

    if first_time_training:
        epsilon = pick_epsilon(played_games, number_of_games)
    else:
        epsilon = 0.00

    prev_state = agent.state(SCREEN_WIDTH, SCREEN_HEIGHT, snake, food)
    prev_pos = snake.head_pos()
    action = pick_action(epsilon, prev_state, agent)
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


def pick_action(epsilon, prev_state, agent):
    if random.uniform(0, 1) < epsilon:
        action = to_categorical(random.randint(0, 3), num_classes=4)
    else:
        prediction = agent.get_model().predict(prev_state.reshape(1, agent.input_neurons))
        action = to_categorical(np.argmax(prediction[0]), num_classes=4)
    return action


def direction(action):
    return map_direction.get(np.argmax(action))


def short_train(agent, snake, food, action, prev_pos, prev_state):
    new_state = agent.state(SCREEN_WIDTH, SCREEN_HEIGHT, snake, food)
    reward = agent.rew(food, prev_pos, snake, prev_state)
    agent.train(prev_state, action, reward, new_state, snake.done)
    agent.remember((prev_state, action, reward, new_state, snake.done))


def print_result(played_games, score):
    print(f'Game: {played_games}\t\tScore {score}')


def prepare_info(scores, record, played_games):
    text_report = ''
    for game, score in enumerate(scores[played_games - save_period:]):
        text_report += f'Game: {played_games - save_period + game + 1}\t\tScore: {score}\n'
    text_report += f'\n\nHighest score: {record}' \
                   f'\nMean: {np.mean(scores[played_games - save_period:])}' \
                   f'\nStd: {np.std(scores[played_games - save_period:])}'
    return text_report


def update_parameters(agent, played_games, snake, scores, record, highlight):
    if training:
        agent.long_train(agent.memory())
    played_games += 1
    score = snake.length() - 1
    scores.append(score)

    if score > record:
        record = score
    if score >= highlight_score_lower_limit:
        add(highlight)



    return played_games, score, record


def save(agent, played_games, scores, record):
    if played_games % save_period == 0 and training:
        agent.get_model().save_weights(weights_path)
        print(f'Model saved at {played_games} games')

        # report to telegram (able to turn it off)
        report(prepare_info(scores, record, played_games))
        remember_highlights()


def final_score(scores, record, played_games):
    print(f'Highest score: {record}\t\t\t', end='')
    print(f'Mean: {np.mean(np.array(scores))}\t\tStd: {np.std(np.array(scores))}')
    report(f'Finished training for {played_games}\n'
           f'Highest score: {record}\n'
           f'Mean: {np.mean(np.array(scores))}\nStd: {np.std(np.array(scores))}')


if __name__ == '__main__':
    main()
