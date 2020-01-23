# Q-network 

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

def one_hot(x):
    return np.identity(16)[x:x+1]

# 입력받을 state의 size, 16개
input_size = env.observation_space.n
# 출력받을 action의 size, 4개
output_size = env.action_space.n
learning_rate = 0.1

X = tf.placeholder(shape = [1,input_size], dtype = tf.float32)
W = tf.Variable(tf.random_uniform([input_size, output_size],0,0.01))

Qpred = tf.matmul(X,W)
Y = tf.placeholder(shape = [1, output_size], dtype = tf.float32)

loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

dis = .99
num_episodes = 2000

rList = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_episodes):
        s = env.reset()
        e = 1./((i/50)+10)
        rAll = 0
        done = False
        local_loss = []
        while not done:
            # Qpred의 X값에 one_hot(s)를 넣어서 학습 시작
            Qs = sess.run(Qpred, feed_dict = {X: one_hot(s)})
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)
            # action한 결과를 각 조건에 대입    
            s1, reward, done, _ = env.step(a)
        
            # Qs 가 2차원 array라서 Qs[0, a]로 출력        
            if done:
                Qs[0, a] = reward
            else:
                # Qs1 : 다음 실행 값
                Qs1 = sess.run(Qpred, feed_dict={ X : one_hot(s1)})
                Qs[0, a] = reward + dis*np.max(Qs1)
        
            # 학습 시작
            sess.run(train, feed_dict = { X:one_hot(s), Y: Qs})
        
            rAll += reward
            s = s1
        rList.append(rAll)
    
print("Percent of successful episodes :" + str(sum(rList)/num_episodes) + "%")
plt.bar(range(len(rList)), rList, color = 'blue')
plt.show()