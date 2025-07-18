import random
import time

class SnakeGame:
    def __init__(self):
        self.direction = (1, 0)
        self.winX = 500
        self.winY = 500
        self.applepos = (-1, -1)
        self.length = 1
        self.snake_body = [(0, 0)]
        self.fps = 2
        self.end = False

    def draw_game(self):
        pass
    
    def move(self):
        if (self.snake_body[0]+self.direction == self.applepos):
            self.snake_body.insert(0, (self.applepos))
            self.apple_creation()
            self.length += 1
        else:
            self.snake_body.pop()
            self.snake_body.insert(0, self.snake_body[0]+self.direction)
    
    def change_dir(self, dir):
        self.direction = dir
    
    def apple_creation(self):
        self.applepos = (random.randint(0, 499), random.randint(0, 499))
    
    def reward(self):
        reward = self.length - (10 if self.end else 0)
        return reward

    def mainloop(self):
        while not self.end:
            self.move()
            time.sleep(1/(self.fps))
