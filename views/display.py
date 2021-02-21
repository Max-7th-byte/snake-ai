import pygame

from views.game import Game
from config import *


def initialize_display(width, height):
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((width, height), 0, 32)
    my_font = pygame.font.SysFont('monospace', font_size, bold=True)
    return clock, screen, my_font


def draw_surface(screen, replaying):
    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()
    game = Game(surface, surface_color1, surface_color2, replaying)
    game.draw_grid()
    return surface, game


def update_display(snake, food, surface, screen, font):
    snake.draw(surface)
    food.draw(surface)
    screen.blit(surface, (0, 0))
    text = font.render(f'Score {snake.score()}', True, text_color, text_background_color)
    screen.blit(text, (5, 10))
    pygame.display.update()
