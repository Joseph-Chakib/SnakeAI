import pygame
from objects import *
from neural import *


SCREEN_WIDTH = 1280
DURATION = 0
SCORE = 0
REWARD = 0
STEP_COUNT = 0
EPISODES = 0
SCREEN_HEIGHT = 720
ROBOT = True
GRID_X = 450
GRID_Y = 150
GRID_LENGTH = 400
FPS = 100


# implement gpu boosting


def initialization(width=SCREEN_WIDTH, height=SCREEN_HEIGHT):
    # Initialize Pygame
    pygame.init()
    clock = pygame.time.Clock()

    if ROBOT:
        brain = Brain(10)
        agent = Agent(brain)

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

    return brain, agent, screen, running, grid, snake, apple, clock


def state_update(snake: Snake, apple: Apple, screen):
    apple.draw(screen)
    snake.move_tail()
    snake.move()
    snake.draw(screen)

def hit_apple(snake: Snake, apple: Apple):
    apple.respawn()
    if snake.get_tail() == []:
        seg = Tail(snake.get_rect().x, snake.get_rect().y)
    else:
        seg = Tail(0, 0)
    snake.grow(seg)

def direction_logic(event, snake: Snake):
    if event.key == pygame.K_w and snake.get_velocity() != (0, 1):
        snake.velocity(0, -1)
    if event.key == pygame.K_s and snake.get_velocity() != (0, -1):
        snake.velocity(0, 1)
    if event.key == pygame.K_a and snake.get_velocity() != (1, 0):
        snake.velocity(-1, 0)
    if event.key == pygame.K_d and snake.get_velocity() != (-1, 0):
        snake.velocity(1, 0)

def display_metrics(score: float, duration: float, reward: float):
    font = pygame.font.Font(None, 36)  # None for default font, 36 is the font size
    score_text = font.render(f"{score}", True, (255, 255, 255))  # True for antialiasing, white color
    screen.blit(score_text, (50, 100))
    duration_text = font.render(f"{duration}", True, (255, 255, 255))  # True for antialiasing, white color
    screen.blit(duration_text, (50, 125))
    reward_text = font.render(f"{reward}", True, (255, 255, 255))  # True for antialiasing, white color
    screen.blit(reward_text, (50, 150))

def game_state(snake: Snake, apple: Apple):
    snake_x, snake_y = snake.get_position()
    apple_x, apple_y = apple.get_position()
    snake_vx, snake_vy = snake.get_velocity()
    right_distance = snake_x - INITIAL_X
    left_distance = GRID_LENGTH - right_distance
    up_distance = snake_y - INITIAL_Y
    down_distance = GRID_LENGTH - up_distance
    # body_distance
    return torch.tensor([snake_x, snake_y, snake_x - apple_x, snake_y - apple_y, snake_vx, snake_vy, right_distance, left_distance, up_distance, down_distance])

def robot_direction_logic(move: torch.tensor, snake: Snake):
    if move == torch.tensor(0) and snake.get_velocity() != (0, 1):
        snake.velocity(0, -1)
    if move == torch.tensor(1) and snake.get_velocity() != (0, -1):
        snake.velocity(0, 1)
    if move == torch.tensor(2) and snake.get_velocity() != (1, 0):
        snake.velocity(-1, 0)
    if move == torch.tensor(3) and snake.get_velocity() != (-1, 0):
        snake.velocity(1, 0)

def step_closer(current: torch.tensor, next: torch.tensor):
    current_proximity_x = abs(current[2].item())
    current_proximity_y = abs(current[3].item())
    next_proximty_x = abs(next[2].item())
    next_proximty_y = abs(next[3].item())

    if next_proximty_x < current_proximity_x or next_proximty_y < current_proximity_y:
        return 2
    else:
        return -3


brain, agent, screen, running, grid, snake, apple, clock = initialization()
hit_apple(snake, apple)
hit_apple(snake, apple)


while running:

    STEP_COUNT += 1

    if STEP_COUNT == steps_per_episode:
        STEP_COUNT = 0
        EPISODES += 1
        collision_experiences, step_experiences, apple_experiences = agent.buffer_size()
        print(f'Experience Data: Collisions: {collision_experiences} | Steps: {step_experiences} | Apples: {apple_experiences}')
        batches = agent.sample()
        agent.train(batches)
        

    if EPISODES == 4000:
        # log average reward
        pass

    if ROBOT:
        current_state = game_state(snake, apple)
        move = agent.decision(current_state)
        robot_direction_logic(move, snake)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and not ROBOT:
            direction_logic(event, snake)

    screen.fill("black")
    grid.draw(screen)
    grid.refresh(GRID_X, GRID_Y)

    state_update(snake, apple, screen)

    # running = not snake.collisions()
    if snake.collisions():

        REWARD = -50
        next_state = game_state(snake, apple)
        agent.store(current_state, move, REWARD, next_state, 1)

        snake, apple, SCORE, DURATION = Snake(), Apple(), 0, 0
        hit_apple(snake, apple)
        hit_apple(snake, apple)
        display_metrics(SCORE, DURATION, REWARD)

    elif snake.get_position() == apple.get_position():

        REWARD = 20
        next_state = game_state(snake, apple)
        agent.store(current_state, move, REWARD, next_state, 0)

        SCORE += 10
        hit_apple(snake, apple)
        display_metrics(SCORE, DURATION, REWARD)

    else:

        next_state = game_state(snake, apple)
        REWARD = 0.1 + step_closer(current_state, next_state)
        agent.store(current_state, move, REWARD, next_state, 0)

        DURATION += 1
        display_metrics(SCORE, DURATION, REWARD)

    pygame.display.flip()
    clock.tick(FPS)
