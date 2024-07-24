import pygame
from objects import *

SCREEN_WIDTH = 1280
DURATION = 0
SCORE = 0
SCREEN_HEIGHT = 720
ROBOT = False
GRID_X = 450
GRID_Y = 150
FPS = 10

# implement gpu boosting


def initialization(width=SCREEN_WIDTH, height=SCREEN_HEIGHT):
    # Initialize Pygame
    pygame.init()
    clock = pygame.time.Clock()

    if ROBOT:
        # initialize a neural network to play
        pass

    # Set up the display
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Snake-AI")

    running = True

    # setup the grid
    grid = Grid(GRID_X, GRID_Y)
    grid.draw(screen)

    # instantiate a snake with direction
    snake = Snake()
    apple = Apple()

    return screen, running, grid, snake, apple, clock

def reset():
    return Snake(), Apple(), 0, 0


def state_update(snake, apple, screen):
    apple.draw(screen)
    snake.move_tail()
    snake.move()
    snake.draw(screen)


def hit_apple(snake, apple):
    apple.respawn()
    if snake.get_tail() == []:
        seg = Tail(snake.get_rect().x, snake.get_rect().y)
    else:
        seg = Tail(0, 0)
    snake.grow(seg)


def direction_logic(event, snake):
    if event.key == pygame.K_w and snake.get_velocity() != (0, 1):
        snake.velocity(0, -1)
    if event.key == pygame.K_s and snake.get_velocity() != (0, -1):
        snake.velocity(0, 1)
    if event.key == pygame.K_a and snake.get_velocity() != (1, 0):
        snake.velocity(-1, 0)
    if event.key == pygame.K_d and snake.get_velocity() != (-1, 0):
        snake.velocity(1, 0)

def display_metrics(score, duration):
    font = pygame.font.Font(None, 36)  # None for default font, 36 is the font size
    score_text = font.render(f"{score}", True, (255, 255, 255))  # True for antialiasing, white color
    screen.blit(score_text, (50, 100))
    duration_text = font.render(f"{duration}", True, (255, 255, 255))  # True for antialiasing, white color
    screen.blit(duration_text, (50, 125))


screen, running, grid, snake, apple, clock = initialization()

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and not ROBOT:
            direction_logic(event, snake)
        else:
            # AI Observe
            # AI Think
            # AI Decide
            pass

    screen.fill("black")
    grid.draw(screen)
    grid.refresh(GRID_X, GRID_Y)

    # running = not snake.collisions()
    if snake.collisions():
        snake, apple, SCORE, DURATION = reset()
        # save_score()
        # store_score()
        # save_duration()
        # store_duration()
        # calculate_fitness()
        # update_brain()
        pass
    if snake.get_position() == apple.get_position():
        hit_apple(snake, apple)
        SCORE +=1

    state_update(snake, apple, screen)
    DURATION += 1 
    display_metrics(SCORE, DURATION)
    pygame.display.flip()
    clock.tick(FPS)
