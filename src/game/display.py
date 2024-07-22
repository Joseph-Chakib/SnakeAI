import pygame
from game.objects import *

def run_game(screen_width=1280, screen_height=720):
    # Initialize Pygame
    pygame.init()
    clock = pygame.time.Clock()

    # Set up the display
    screen = pygame.display.set_mode((screen_width, screen_height))
    running = True

    #setup the grid
    grid = Grid(450, 150)
    grid.draw(screen)
    cordinates = grid.cordinates

    #instantiate a snake
    snake = Snake(cordinates)

    #instantiate an apple in the middle of the board

    # Main loop
    while running:
        # Poll for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fill the screen with a color to wipe away anything from last frame
        screen.fill("black")

        # Screen Logic
        grid.draw(screen)
        grid.refresh(450, 150)

        # Snake Logic
        # check if collisions occur
            # If apple collision
            # snake grow
            # apple respawn
            # if wall collision
            # quit
        snake.draw(screen)
        # change direction of movement for key press events
            # if right then snake.move
        snake.move()


        # Apple Logic

       # Flip the display to put your work on screen
        pygame.display.flip()

        # Set frame rate
        clock.tick(8)

    #quite after game stops running
    # pygame.quit()

# Example call to the function
if __name__ == "__main__":
    run_game()
