# is_slippery = True인 환경에서 새로운 Q-learning알고리즘 이용 -> 기존 알고리즘보다 더 잘나옴
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

# 이 환경에서 is_slippery = True 여서 불러오면 됨
env = gym.make('FrozenLake-v0')

# Q[state, action]을 불러오고 초기화
Q = np.zeros([env.observation_space.n,env.action_space.n])

learning_rate = .85
dis = .99
num_episodes = 2000

rList = []

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    # done : hole이나 goal에 도달하면 true로 반환되고, 게임이 끝남(1회)
    done = False
    
    # done이 될 때 까지 실행
    while not done:
        # action을 결정하기 위해 만든 식, e-greedy, noise 방식이 있는데 여기선 noise 수행
        # noise : 기존 q 값에 랜덤 값을 더해서 선택하는 방식
        action = np.argmax(Q[state,:] + np.random.randn(1, env.action_space.n)/(i+1))
        # env.step(action) : action을 취하면 다음 state, 받은 reward를 얻음
        new_state, reward, done, _ = env.step(action)
        
        
        # update Q-table, learning_rate를 추가함으로 is_slippery = True 환경에서도 잘 이용됨
        Q[state,action] = (1-learning_rate)*Q[state, action] + learning_rate*(reward + dis*np.max(Q[new_state,:]))
        
        rAll += reward
        state = new_state
    rList.append(rAll)
    
print("Success rate:" + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color = 'blue')
plt.show()
        