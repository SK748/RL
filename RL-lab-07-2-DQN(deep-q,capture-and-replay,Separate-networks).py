# use deep-q, capture-and-replay, separate networks

import numpy as np
import tensorflow as tf
import random
import dqn
from collections import deque

import gym
env = gym.make('CartPole-v0')

input_size = env.observation_space.shape[0]
output_size = env.action_space.n

dis = 0.9
REPLAY_MEMORY = 50000

def simple_replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)
    
    # train_batch에 있는 정보들을 얻어와 DQN을 학습시키고, update
    for state, action, reward, next_state, done in train_batch:
        # DQN에 state를 넣어 현재 Q의 추정값을 가져온다.
        Q = mainDQN.predict(state)
        
        if done:
            Q[0, action] = reward
        else:
            # Q의 값이 update 되면 next_state도 update를 방지하고자, target은 update 하지 않음
            Q[0, action] = reward + dis*np.max(targetDQN.predict(next_state))
        
        # 랜덤하게 뽑아온 값들을 Q를 통해 학습시킨뒤 DQN update
        x_stack = np.vstack([x_stack, state])
        y_stack = np.vstack([y_stack, Q])
        
    return mainDQN.update(x_stack, y_stack)

def bot_play(mainDQN):
    # 내가 만든 것을 실행시키는 함수
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        a = np.argmax(mainDQN.predict(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print("Total score : {}".format(reward_sum))
            break

            
            
def get_copy_var_ops(*, dest_scope_name = "target", src_scope_name = "main"):
    # target의 Q와 main의 Q를 받아 같게 만들어 주는 함수
    op_holder = []
    
    
    # SCOPE에 있는 값 중 바뀌는 것 만 가져옴
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = dest_scope_name)
    
    
    for src_var, dest_var in zip(src_vars, dest_vars):
        # dest_var(target의 값)을 src_var(main의 값)으로 바꿈
        op_holder.append(dest_var.assign(src_var.value()))
        
    return op_holder
            
def main():
    max_episodes = 5000
    
    # store the previous observations in replay memory  저장시키고, 랜덤하게 꺼내기 위해
    replay_buffer = deque()
    
    with tf.Session() as sess:
        # mainDQN : WX, targetDQN : Y 값
        mainDQN = dqn.DQN(sess, input_size, output_size, name = "main")
        targetDQN = dqn.DQN(sess, input_size, output_size, name = "target")
        tf.global_variables_initializer().run()
        
        # Qpred와 Target의 값을 같게 하는 함수 , Qpred = WX라고 생각
        copy_ops = get_copy_var_ops(dest_scope_name = "target", src_scope_name = "main")
        # 시작
        sess.run(copy_ops)
        
        for episodes in range(max_episodes):
            e = 1./((episodes / 10) +1)
            step_count = 0
            done = False
            
            state = env.reset()
            
            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))
                    
                next_state, reward, done, _ = env.step(action)
                if done:
                    reward = -100
                
                replay_buffer.append((state, action, reward, next_state, done))
                # replay_buffer가 너무 커지는 것을 방지 -> 일정 수 보다 커지면 가장 먼저 받은 값을 내보냄
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()
                   
                state = next_state
                step_count += 1
                if step_count > 10000:
                    break
                    
            print("Episodes : {} steps : {}".format(episodes, step_count))
            if step_count > 10000:
                pass
            # break
            
            # 10번 반복할때마다 모아놓은 replay_buffer에서 random으로 값을 추출하고, 학습시켜 Q_pred를 update 함
            if episodes % 10 == 1:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = simple_replay_train(mainDQN, targetDQN, minibatch)
                print("Loss :",loss)
                
                # target을 고정시킨 상태에서 
                sess.run(copy_ops)
        bot_play(mainDQN)                
                
if __name__ =="__main__":
    main()