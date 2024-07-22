import pygame
import random

class Grid:
    def __init__(self, x, y, cordinates=[], SIZE=20, color=(255, 255, 255)) -> None:
        self.rect = pygame.Rect(x, y, SIZE, SIZE)
        self.color = color
        self.size = SIZE
        # self.cordinates = cordinates
        return None
    def draw(self, screen) -> None:
        for i in range(20):
            self.rect.y = 150
            self.rect.x += self.size
            for j in range(20):
                self.rect.y += self.size
                pygame.draw.rect(screen, self.color, self.rect, 1)
                # self.cordinates.append([self.rect.x, self.rect.y])
                # print(f"Block Position: ({self.rect.x}, {self.rect.y})")
        # print(f"Length is {len(self.cordinates)} \n")
        # print(self.cordinates)
        # print('\n')
        return None
    def refresh(self, x, y) -> None:
        # self.cordinates = []
        self.rect.x = x
        self.rect.y = y
        return None
    def cordinates(self) -> list:
        #return self.cordinates
        pass

class Snake:
    def __init__(self, cordinates, SIZE=20, color=(0, 255, 0)) -> None:
        try:
            assert len(cordinates) == 200
        except:
            # print(cordinates)
            # print(len(cordinates))
            pass
        self.cordinates = cordinates
        self.rect = pygame.Rect(cordinates[200][0], cordinates[200][1], SIZE, SIZE)  # Define the player as a rectangle
        self.color = color  # Set the color of the player
        self.size = SIZE
        self.body = []

    def move(self):
        self.rect.x += 20  # Move left

    def grow(self):
        # add additional unit to the end of the snake body
        pass

    def collisions():
        pass

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)  # Draw the player on the screen

class Apple:
    def __init__(self, x, y, SIZE=20, color=(255, 0, 0)) -> None:
        self.rect = pygame.Rect(x, y, SIZE, SIZE)
        self.color = color
    def respawn(self):
        #set x, y cordinates to a random number between 0 and 19
        #set x and y to start_x, start_y + 20*x or 20*y
        pass
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)  # Draw the player on the screen

#movement logic
    #snake starts with movement vector (5, 0)
    #what is an elegent way to say that we are not allowed to flip position i.e cannot go up then down?
    #we could just save a state representaiton and then work off of that as a direction string
    #The move function is continously called however and a key press only changes that direction
    #If up then (0, 1)
    #if down then (0, -1)
    #if left then (-1, 0)
    # if right the (1, 0)


# How do i make it so that abjects only move within the confines of the grid?
# Save rect cordinates as an array to move through 