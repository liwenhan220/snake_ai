from snake_game import snake_game
env = snake_game()
from keras.models import load_model, Sequential
from keras.layers import *
from keras.optimizers import RMSprop
import numpy as np
import cv2
from getkeys import key_check

MODEL = 'snake-best.model'
EPISODES = 10000

network = load_model(MODEL)
def main():

    for episode in range(EPISODES):
        current_state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            keys = key_check()
            if 'W' in keys:
                action = 0
            elif 'S' in keys:
                action = 1
            elif 'A' in keys:
                action = 2
            elif 'D' in keys:
                action = 3
            else:
                qs = network.predict(np.array(current_state).reshape(1, *env.observation_space)/255.0)[0]
                action = np.argmax(qs)
            new_state, reward, done = env.step(action)
            env.render()
            current_state = new_state
            ep_reward += reward

if __name__ == '__main__':
    main()
