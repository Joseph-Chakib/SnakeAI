import pygame
from objects import *
from neural import *


SCREEN_WIDTH = 1280
DURATION = 0
SCORE = 0
REWARD = 0
COLLISIONS = 0
SCREEN_HEIGHT = 720
ROBOT = True
GRID_X = 450
GRID_Y = 150
GRID_LENGTH = 400
FPS = 60

# implement gpu boosting


def initialization(width=SCREEN_WIDTH, height=SCREEN_HEIGHT):
    # Initialize Pygame
    pygame.init()
    clock = pygame.time.Clock()

    if ROBOT:
        brain = Brain(10)
        agent = Agent(brain)
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

def display_metrics(score: int, duration: int, reward: int):
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
    return torch.tensor([snake_x, snake_y, apple_x, apple_y, snake_vx, snake_vy, right_distance, left_distance, up_distance, down_distance])

def robot_direction_logic(move: torch.tensor, snake: Snake):
    if move == torch.tensor(0) and snake.get_velocity() != (0, 1):
        snake.velocity(0, -1)
    if move == torch.tensor(1) and snake.get_velocity() != (0, -1):
        snake.velocity(0, 1)
    if move == torch.tensor(2) and snake.get_velocity() != (1, 0):
        snake.velocity(-1, 0)
    if move == torch.tensor(3) and snake.get_velocity() != (-1, 0):
        snake.velocity(1, 0)    

brain, agent, screen, running, grid, snake, apple, clock = initialization()

while running:

    if ROBOT:
        current_state = game_state(snake, apple)
        # print(f'Current State: {state}')
        move = agent.decision(current_state)
        number = random.randint(0, 3)
        move = torch.tensor(number)
        robot_direction_logic(move, snake)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and not ROBOT:
            direction_logic(event, snake)

    screen.fill("black")
    grid.draw(screen)
    grid.refresh(GRID_X, GRID_Y)

    # running = not snake.collisions()
    if snake.collisions():
        COLLISIONS += 1
        REWARD = DURATION + SCORE*50 - COLLISIONS*100
        
        snake, apple, SCORE, DURATION, COLLISIONS = Snake(), Apple(), 0, 0, 0

    if snake.get_position() == apple.get_position():
        hit_apple(snake, apple)
        SCORE +=1

    state_update(snake, apple, screen)

    DURATION += 1 
    REWARD = DURATION + SCORE*50 - COLLISIONS*100

    display_metrics(SCORE, DURATION, REWARD)

    pygame.display.flip()
    clock.tick(FPS)
