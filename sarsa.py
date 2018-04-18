#!/usr/bin/python
# -*- coding: UTF-8 -*- 

import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
from maze import * 

def set_rand_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

class SarsaTable(object):

    def __init__(self, actions):
        self.table = pd.DataFrame(
            columns = actions
            , dtype=np.float64
            )
        self.actions = actions

    def choose_action(self, state, epsilon = 0.9):
        self.check_state_exist(state)
        if np.random.uniform() > epsilon:
            return np.random.choice(self.actions)
        else:
            state_action = self.table.loc[state, :]
            #同一个 state, 可能会有多个相同的 Q action value, 所以我们乱序一下
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            return state_action.idxmax()

    def learn(self, cur_state, action, next_state, reward, next_action, learning_rate = 0.01, reward_decent_rate = 0.9):
        predict_value = self.table.loc[cur_state, action]
        if next_state is None:
            target = reward
        else:
            self.check_state_exist(next_state)
            target = reward + reward_decent_rate * self.table.loc[next_state, next_action]
        self.table.loc[cur_state, action] += learning_rate * (target - predict_value)

    def check_state_exist(self, state):
        if state not in self.table.index:
            self.table = self.table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index = self.table.columns,
                    name = state,
                )
            )



def play(env, rl):
    env.reset_env()
    is_terminate = False
    while not is_terminate:
        s = env.get_state()
        action = rl.choose_action(s, 1.0)
        print str(env)
        print "action: " + action
        s, is_terminate , _ = env.do_action(action)
        time.sleep(1)


if __name__ == '__main__':
    set_rand_seed(7)
    # 不好搞定
    env = MazeEnv(20, 30, block_rate = 0.25)
    #env = MazeEnv(19, 19, block_rate = 0.25)
    rl = SarsaTable(env.get_all_actions())
    epoch_count = 1000
    avg_rate = []
    avg_len = []
    all_reward = 0
    all_step = 0
    cnt = 0
    for epoch in range(epoch_count):
        env.reset_env()
        s = env.get_state()
        action = rl.choose_action(s)
        is_terminate = False
        step_list = []
        while not is_terminate:
            #print "action: " + str(action)
            (ns, is_terminate, reward) = env.do_action(action)
            n_act = rl.choose_action(ns)
            # 使用这种更新效率比较底下
            #rl.learn(s, action, ns, reward, n_act)
            step_list.append((s, action, ns, reward, n_act))
            s = ns
            action = n_act
        for i in range(len(step_list) - 1, -1, -1):
            s, action, ns, _, n_act = step_list[i]
            rl.learn(s, action, ns, reward / float(len(step_list) - i) , n_act)
        all_reward += reward
        all_step += len(step_list)
        cnt += 1
        if epoch % (epoch_count / 100) == 0:
            avg_r = all_reward / float(cnt)
            avg_s = all_step / float(cnt)
            all_reward = 0
            all_step = 0
            cnt = 0
            avg_rate.append(avg_r)
            avg_len.append(avg_s)
            print "epoch %d avg_s [%0.4lf] avg_r [%0.4lf]" % (epoch, avg_s, avg_r)
    env.reset_env()
    print "------------"
    print str(env)
    print "------------"
    play(env, rl)
    #print "sarsa_tables:"
    #print rl.table
    f = plt.figure()
    plt_1 = f.add_subplot(111)
    plt_1.plot(range(len(avg_rate)), avg_rate, 'r', label="average reward")
    plt_2 = plt_1.twinx()
    plt_2.plot(range(len(avg_len)), avg_len, 'b--', label="average steps")
    plt.legend()
    plt.show()



