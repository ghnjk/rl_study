#!/usr/bin/python
# -*- coding: UTF-8 -*-
import gym
from dqn import *

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = env.unwrapped  # 不做这个会有很多限制
    rl = DeepQNetwork(feature_cnt=env.observation_space.shape[0],
                      action_cnt=env.action_space.n,
                      learning_rate=0.01,
                      upgrade_net_iter=100,
                      max_epsilon=0.9,
                      epsilon_increasement=0.0008,
                      memory_size=2000)
    step_count = 0
    for epoch in range(100):
        observation = env.reset()
        ep_r = 0
        while True:
            env.render()  # 刷新环境
            action = rl.choose_action(observation)
            next_observation, reward, is_done, info = env.step(action)
            x, x_dot, theta, theta_dot = next_observation
            # x 是车的水平位移, 所以 r1 是车越偏离中心, 分越少
            # theta 是棒子离垂直的角度, 角度越大, 越不垂直. 所以 r2 是棒越垂直, 分越高
            r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
            reward = r1 + r2   # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率
            rl.store_train(observation, action, reward, next_observation)
            if step_count > 1000:
                rl.learn()
            ep_r += reward
            if is_done:
                print('epoch: ', epoch,
                      '| ep_r: ', round(ep_r, 2),
                      '| epsilon: ', round(rl.epsilon, 2))
                break
            observation = next_observation
            step_count += 1
    rl.plot_loss()
