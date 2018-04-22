#!/usr/bin/python
# -*- coding: UTF-8 -*-
import gym
from dqn import *

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    rl = DeepQNetwork(feature_cnt=env.observation_space.shape[0],
                      action_cnt=env.action_space.n,
                      learning_rate=0.001,
                      upgrade_net_iter=300,
                      max_epsilon=0.9,
                      epsilon_increasement=0.0001,
                      memory_size=3000)
    step_count = 0
    for epoch in range(10):
        observation = env.reset()
        ep_r = 0
        while True:
            env.render()
            action = rl.choose_action(observation)
            next_observation, reward, is_done, info = env.step(action)
            position, velocity = next_observation
            reward = abs(position - (-0.5))
            rl.store_train(observation, action, reward, next_observation)
            if step_count > 1000:
                rl.learn()
            ep_r += reward
            if is_done:
                get = '| Get' if next_observation[0] >= env.unwrapped.goal_position else '| ----'
                print('epoch: ', epoch,
                      get,
                      ' ep_r: ', round(ep_r, 4),
                      '| epsilon: ', round(rl.epsilon, 2))
                break
            observation = next_observation
            step_count += 1
    rl.plot_loss()
