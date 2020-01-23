# Q-network 

import gym
import numpy as np
import tensorflow as tf

env = gym.make('CartPole-v0')

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
learning_rate = 0.1e-1

X = tf.placeholder(tf.float32, shape = [1,input_size], name = 'input_x')
W1 = tf.get_variable('W1', shape = [input_size, output_size], initializer = tf.contrib.layers.xavier_initializer())

Qpred = tf.matmul(X,W1)
Y = tf.placeholder(shape = [None, output_size], dtype = tf.float32)

loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

dis = .99
num_episodes = 2000
rList = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_episodes):
        s = env.reset()
        e = 1./((i/10)+1)
        rAll = 0
        done = False
        step_count = 0
        
        while not done:
            
            step_count += 1
            # one_hot을 쓸 수 없어 reshape 사용
            x = np.reshape(s, [1, input_size])
            
            Qs = sess.run(Qpred, feed_dict = {X : x})
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)
            # action한 결과를 각 조건에 대입    
            s1, reward, done, _ = env.step(a)
        
            # Qs 가 2차원 array라서 Qs[0, a]로 출력        
            if done:
                Qs[0, a] = -100
            else:
                # Qs1 : 다음 실행 값
                x1 = np.reshape(s1, [1, input_size])
                Qs1 = sess.run(Qpred, feed_dict={ X : x1})
                Qs[0, a] = reward + dis*np.max(Qs1)
        
            # 학습 시작
            sess.run(train, feed_dict = { X: x, Y: Qs})
            s = s1
            
        rList.append(step_count)
        print("Episode : {} steps : {}".format(i, step_count))
        
        if len(rList) > 10 and np.mean(rList[-10:]) > 500:
            break
            
# See our trained network in action
observation = env.reset()
reward_sum = 0
while True:
    env.render()
    
    x = np.reshape(observation, [1, input_size])
    Qs = sess.run(Qpred, feed_dict = {X : x})
    a = np.argmax(Qs)
    
    observation, reward, done, _ = env.step(a)
    reward_sum += reward
    if done:
        print("Total score : {}".format(reward_sum))
        break