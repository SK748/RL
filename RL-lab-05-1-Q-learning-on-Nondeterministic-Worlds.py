# is_slippery = True 즉, 내 마음대로 되지 않는 환경


import readchar
import gym
import numpy as np
from gym.envs.registration import register
import random as pr

# 이 환경에서 is_slippery = True 여서 불러오면 됨
env = gym.make('FrozenLake-v0')

env.render()

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = { '\x1b[A' : UP, '\x1b[B' : DOWN, '\x1b[C' : RIGHT, '\x1b[D' : LEFT }

while True:
    key = readchar.readkey()
    if key not in arrow_keys.keys():
        print("Game aborted!")
        break
    
    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render()
    print("State :",state,"Action :",action, "Reward :", reward, "Info :", info)
    
    if done:
        print("Finished with reward",reward)
        break