import numpy as np
import cv2
from collections import deque
import time
import math

class snake_game:
    def __init__(self):
        self.size = 20
        self.observation_space = (20,20,1)
        self.subject = (255, 0, 0)
        self.food = (0, 0, 125)
        self.segment = (255,255,255)
        self.wall = (255,255,255)
        self.total_reward = 0
        self.action_space = 4

    def init_game(self):
        self.img = np.zeros((self.size,self.size,3))
        for x in range(len(self.img)):
            self.img[x][0] = self.wall
            self.img[x][self.size-1] = self.wall
        for y in range(self.size):
            self.img[0][y] = self.wall
            self.img[self.size-1][y] = self.wall
        return self.img
    
    def reset(self):
        self.sub_x = int(self.size/2)
        self.sub_y = int(self.size/2)
        self.food_x = np.random.randint(2,self.size-2)
        self.food_y = np.random.randint(2,self.size-2)        
        self.img = self.init_game()
        self.img[self.sub_x][self.sub_y] = self.subject
        self.img[self.food_x][self.food_y] = self.food
        self.reward = 0
        self.total_reward = 0
        self.segments = []
        self.last_n = 5
        
        state = []
        
        for x in range(self.sub_x-10,self.sub_x+10):
            ys = []
            for y in range(self.sub_y-10,self.sub_y+10):
                try:
                    ys.append(np.mean(self.img[x][y]))
                except:
                    ys.append(255)
            state.append(ys)
                
        return np.array(state)
                
    def update_frame(self):
        self.img = np.zeros((self.size,self.size,3))
        self.img = self.init_game()
        self.img[self.sub_x][self.sub_y] = self.subject
        self.img[self.food_x][self.food_y] = self.food
        try:
            for i in range(len(self.segments)):
                self.img[self.segments[i][0]][self.segments[i][1]] = self.segment
        except:
            pass
                                              
        return self.img

    def render(self):
        cv2.imshow('snake_game',cv2.resize(self.img,(500,500)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    def step(self,input_n):
        if self.last_n == 0 and input_n == 1 or self.last_n == 2 and input_n == 3 or self.last_n == 1 and input_n == 0 or self.last_n == 3 and input_n == 2:
                input_n = self.last_n

        self.last_n = input_n
        # moving up
        if input_n == 0:
            self.sub_x -= 1
            
        # moving down
        elif input_n == 1:
            self.sub_x += 1
            
        # moving left
        elif input_n == 2:
            self.sub_y -= 1
            
        # moving right
        elif input_n == 3:
            self.sub_y += 1
            
##        if self.sub_x > self.size-1 or self.sub_x < 0 or self.sub_y < 0 or self.sub_y > self.size-1:
##            self.terminal = True
##            self.reward = -1

        if all(self.img[self.sub_x][self.sub_y] == self.wall):
            self.terminal = True
            self.reward = -1
            
        elif math.sqrt((self.food_x - self.sub_x)**2) <= 1 and math.sqrt((self.food_y - self.sub_y)**2) <= 1:
            self.terminal = False
            self.reward = 1
            self.food_x = np.random.randint(2,self.size-2)
            self.food_y = np.random.randint(2,self.size-2)
            self.img = self.update_frame()    
        else:
            self.terminal = False
            self.reward = 0
            self.img = self.update_frame()

        self.segments.append([self.sub_x,self.sub_y])
        if len(self.segments) > self.total_reward + 3:
            del self.segments[:(len(self.segments)-self.total_reward-3)]
    

        state = []
        
        for x in range(self.sub_x-10,self.sub_x+10):
            ys = []
            for y in range(self.sub_y-10,self.sub_y+10):
                try:
                    ys.append(np.mean(self.img[x][y]))
                except:
                    ys.append(255)
            state.append(ys)

        if self.reward == 1:
            self.total_reward += self.reward

        return np.array(state), self.reward, self.terminal


