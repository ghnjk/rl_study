#!/usr/bin/python
# -*- coding: UTF-8 -*-
import gym
from dqn import *
ACTION_SPACE = 11    # 将原本的连续动作分离成 11 个动作

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    env.seed(1)
    print('gym env: action_box (low: ', env.action_space.low,
          ', high: ', env.action_space.high,
          'feature_cnt: ', env.observation_space.shape[0])
    print('observation: ', env.observation_space.shape)
    rl = DeepQNetwork(feature_cnt=env.observation_space.shape[0],
                      action_cnt=ACTION_SPACE,
                      learning_rate=0.001,
                      upgrade_net_iter=300,
                      max_epsilon=0.9,
                      epsilon_increasement=0.0001,
                      memory_size=3000)
    step_count = 0
    for epoch in range(50):
        observation = env.reset()
        ep_r = 0
        while True:
            env.render()
            action = rl.choose_action(observation)
            f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # 在 [-2 ~ 2] 内离散化动作
            next_observation, reward, is_done, info = env.step([f_action])
            reward /= 10
            rl.store_train(observation, action, reward, next_observation)
            if step_count > rl.memory_size:
                rl.learn()
            ep_r += reward
            if is_done:
                print('epoch: ', epoch,
                      ' ep_r: ', round(ep_r, 4),
                      '| epsilon: ', round(rl.epsilon, 2),
                      '| step_count: ', step_count)
                break
            observation = next_observation
            step_count += 1
    import matplotlib.pyplot as plt
    plt.plot(np.array(rl.history_q), label = 'q')
    plt.grid()
    plt.xlabel('step')
    plt.ylabel('q eval')
    plt.show()
    rl.plot_loss()
