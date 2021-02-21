import pickle as pkl
import time

from views.display import *
from views.game import Snake, Food

highlights = list()


class Highlight(object):

    def __init__(self):
        self._score = 0
        self._states = list()


    def replay(self):
        clock, screen, font = initialize_display(SCREEN_WIDTH, SCREEN_HEIGHT)
        surface, game = draw_surface(screen, True)
        snake = Snake(snake_color, game)
        Food(food_color, snake, position=self._states[0].position())
        for direction, food, next_pos in self._states[1:]:
            clock.tick(speed)
            snake.turn(direction, food, next_position=next_pos)
            update_display(snake, food, surface, screen, font)
        time.sleep(5)

    def remember(self, state):
        self._states.append(state)


def add(highlight):
    highlights.append(highlight)


def remember_highlights():
    with open(highlights_path, 'wb') as f:
        pkl.dump(highlights, f)


def get_highlights():
    global highlights
    with open(highlights_path, 'rb') as f:
        highlights = pkl.load(f)
    return highlights
