import pygame
import random

SIZE = 20  # block size
INITIAL_X = 450
INITIAL_Y = 150


class Grid:

    def __init__(self, x=INITIAL_X, y=INITIAL_Y, color=(255, 255, 255)) -> None:
        self.rect = pygame.Rect(x, y, SIZE, SIZE)
        self.color = color
        self.size = SIZE
        return None

    def draw(self, screen) -> None:
        for i in range(20):
            self.rect.y = INITIAL_Y
            self.rect.x += self.size
            for j in range(20):
                self.rect.y += self.size
                pygame.draw.rect(screen, self.color, self.rect, 1)
        return None

    def refresh(self, x, y) -> None:
        # self.cordinates = []
        self.rect.x = x
        self.rect.y = y
        return None


class Snake:

    def __init__(self, color=(0, 255, 0)) -> None:
        # Define the player as a rectangle
        self.rect = pygame.Rect(650, 350, SIZE, SIZE)
        self.color = color  # Set the color of the player
        self.size = SIZE
        self.speed = pygame.Vector2(0, 1)
        self.tail = []

    def move(self):
        self.rect.x += self.speed.x * SIZE
        self.rect.y += self.speed.y * SIZE

    def velocity(self, x, y):
        self.speed.x = x
        self.speed.y = y

    def get_rect(self):
        return self.rect

    def grow(self, seg):
        self.tail.append(seg)
        pass

    def move_tail(self):
        
        if not self.tail:
            return None

        for i in range(len(self.tail)-1, 0, -1):
            self.tail[i].move(self.tail[i-1].get_rect().x,
                              self.tail[i-1].get_rect().y)

        self.tail[0].move(self.rect.x, self.rect.y)

    def get_tail(self):
        return self.tail

    def collisions(self):
        for seg in self.tail:
            if (self.rect.x, self.rect.y) == (seg.get_rect().x, seg.get_rect().y):
                return True

        if self.rect.x < INITIAL_X + 20 or self.rect.x > INITIAL_X + 400:
            return True

        if self.rect.y < INITIAL_Y + 20 or self.rect.y > INITIAL_Y + 400:
            return True

        return False

    def draw(self, screen):
        # Draw the player on the screen
        pygame.draw.rect(screen, self.color, self.rect)
        for seg in self.tail:
            # add functionality for drawing the body of the snake
            pygame.draw.rect(screen, self.color, seg)

    def get_position(self):
        return self.rect.x, self.rect.y

    def get_velocity(self):
        return (self.speed.x, self.speed.y)


class Tail:
    def __init__(self, x, y) -> None:
        self.rect = pygame.Rect(x, y, SIZE, SIZE)
        pass

    def get_rect(self):
        return self.rect

    def move(self, x, y):
        self.rect.x = x
        self.rect.y = y


class Apple:

    def __init__(self, color=(255, 0, 0)) -> None:
        self.rect = pygame.Rect(750, 250, SIZE, SIZE)
        self.color = color
        pass

    def respawn(self):
        # TODO: Fix respawn mechanic so that apple does not spawn in a location with a tail body present
        x = random.randint(1, 19)
        y = random.randint(1, 19)
        self.rect.x = (20 * x) + INITIAL_X
        self.rect.y = (20 * y) + INITIAL_Y
        pass

    def draw(self, screen):
        # Draw the apple on the screen
        pygame.draw.rect(screen, self.color, self.rect)

    def get_position(self):
        return self.rect.x, self.rect.y
