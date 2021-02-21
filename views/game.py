import sys
import random

import pygame as p

from config import *


class Game:

    def __init__(self, surface, color1, color2, replaying):
        self._surface = surface
        self._color1 = color1
        self._color2 = color2
        self._replaying = replaying


    def draw_grid(self):
        for y in range(0, int(GRID_HEIGHT)):
            for x in range(0, int(GRID_WIDTH)):
                if (x + y) % 2 == 0:
                    rect = p.Rect((x * GRID_SIZE, y * GRID_SIZE), (GRID_SIZE, GRID_SIZE))
                    p.draw.rect(self._surface, self._color1, rect)
                else:
                    rect1 = p.Rect((x * GRID_SIZE, y * GRID_SIZE), (GRID_SIZE, GRID_SIZE))
                    p.draw.rect(self._surface, self._color2, rect1)

    def replaying(self):
        return self._replaying


class Snake:

    def __init__(self, color, game):
        self._color = color
        self._game = game

        self._length = 1
        self._positions = [((SCREEN_WIDTH / 2), (SCREEN_HEIGHT / 2))]
        self._direction = random.choice([LEFT, UP, RIGHT, DOWN])
        self._score = 0
        self._done = False
        self._eaten = False

        self.init_moving()
        self.update_moving()


    def update_moving(self):
        self.init_moving()
        if self._direction == UP:
            self.moving_up = True
        elif self._direction == DOWN:
            self.moving_down = True
        elif self._direction == RIGHT:
            self.moving_right = True
        elif self._direction == LEFT:
            self.moving_left = True


    def init_moving(self):
        self.moving_left = False
        self.moving_right = False
        self.moving_up = False
        self.moving_down = False


    def turn(self, direction, food, next_position=(0, 0)):
        if self._length > 1 and (direction[0] * -1, direction[1] * -1) == self._direction:
            self._done = True
        else:
            self._direction = direction
            self.update_moving()
        if self._game is not None:
            self._game.draw_grid()
        self.move(food, next_position)


    def move(self, food, next_position=(0, 0)):
        self.eat(food, next_position)
        x, y = self._direction
        new_pos = self.new(x, y)
        if self.failed(new_pos):
            self._done = True
        else:
            self._positions.insert(0, new_pos)
            if len(self._positions) > self._length:
                self._positions.pop()


    def eat(self, food, next_position=(0, 0)):
        if self.head_pos() == food.position():
            self._length += 1
            self._score += 1
            food.spawn(self, self._game, next_position)
            self._eaten = True
        else:
            self._eaten = False


    def draw(self, surface):
        for pos in self._positions:
            r = p.Rect((pos[0], pos[1]), (GRID_SIZE, GRID_SIZE))
            p.draw.rect(surface, self._color, r)


    def handle_keys(self, food):
        executed_turn = False
        for event in p.event.get():
            if event.type == p.QUIT:
                p.quit()
                sys.exit()
            elif event.type == p.KEYDOWN:
                executed_turn = True
                if event.key == p.K_w:
                    self.turn(UP, food)
                elif event.key == p.K_s:
                    self.turn(DOWN, food)
                elif event.key == p.K_d:
                    self.turn(RIGHT, food)
                elif event.key == p.K_a:
                    self.turn(LEFT, food)

        if not executed_turn:
            self.turn(self._direction, food)





    def reset(self):
        self.length = 1
        self.positions = [((SCREEN_WIDTH / 2), (SCREEN_HEIGHT / 2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.score = 0
        self.done = False
        self.eaten = False


    def head_pos(self):
        return self._positions[0]


    def new(self, x, y):
        return (((self.head_pos()[0] + (x * GRID_SIZE)) % SCREEN_WIDTH),
                (self.head_pos()[1] + (y * GRID_SIZE)) % SCREEN_HEIGHT)


    @staticmethod
    def check_exit(event):
        if event.type == p.QUIT:
            p.quit()
            sys.exit()



    def is_turned_into_itself(self, new):
        return len(self._positions) > 2 and new in self._positions[2:]


    def ran_into_x_border(self):
        x, _ = tuple(self.head_pos())
        return (x == 0 and self.moving_left) or \
               (x == SCREEN_WIDTH - GRID_SIZE and self.moving_right)
    
    
    def ran_into_y_border(self):
        _, y = tuple(self.head_pos())
        return (y == 0 and self.moving_up) or \
               (y == SCREEN_HEIGHT - GRID_SIZE and self.moving_down)
    
    
    def failed(self, direction):
        return \
            self.is_turned_into_itself(direction) or \
            self.ran_into_x_border() or \
            self.ran_into_y_border()


    def positions(self):
        return self._positions

    def eaten(self):
        return self._eaten

    def done(self):
        return self._done

    def score(self):
        return self._score

    def length(self):
        return self._length

    def game(self):
        return self._game



class Food:

    def __init__(self, color, snake, position=(0, 0)):
        self._position = (0, 0)
        self._color = color
        self.spawn(snake, snake.game(), position)

    def spawn(self, snake, game, position):
        if game is None or not game.replaying():
            food_pos = Food.random_pos()
            while food_pos in snake.positions():
                food_pos = Food.random_pos()
            self._position = food_pos
        else:
            self._position = position


    def draw(self, surface):
        r = p.Rect((self._position[0], self._position[1]), (GRID_SIZE, GRID_SIZE))
        p.draw.rect(surface, self._color, r)
        p.draw.rect(surface, self._color, r, 1)


    @staticmethod
    def random_pos():
        return (random.randint(0, int(GRID_WIDTH - 1)) * GRID_SIZE,
                random.randint(0, int(GRID_HEIGHT - 1)) * GRID_SIZE)


    def position(self):
        return self._position
