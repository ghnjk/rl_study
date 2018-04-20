#!/usr/bin/python
# -*- coding: UTF-8 -*- 

import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from maze import * 
from collections import deque

def set_rand_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class DeepQNetwork(object):

    def __init__(self, feature_cnt, action_cnt
        , upgrade_net_iter
        , batch_size = 32
        , memory_size = 2000
        , reward_decent_rate = 0.9
        , learning_rate = 0.01
        , epsilon_increasement = 0
        , max_epsilon = 0.9):
        # 配置
        self.feature_cnt = feature_cnt
        self.action_cnt = action_cnt
        self.reward_decent_rate = reward_decent_rate
        self.learning_rate = learning_rate
        self.upgrade_net_iter = upgrade_net_iter # 更新网络的间隔
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.max_epsilon = max_epsilon
        self.epsilon_increasement = epsilon_increasement
        self.epsilon = 0 if self.epsilon_increasement != 0 else self.max_epsilon
        # 临时
        self.memory = deque(maxlen=self.memory_size)
        self.learn_count = 0

        # 建立网络
        self.build_net()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.history_loss = []

    def choose_action(self, cur_state):
        if np.random.uniform() < self.epsilon:
            cur_state = cur_state
            action_values = self.session.run(self.q_eval
                , feed_dict = {
                    self.cur_state : [cur_state]
                })
            action = np.argmax(action_values)
        else:
            action = np.random.randint(0, self.action_cnt)
        return action

    def store_train(self, cur_state, action, reward, next_state):
        self.memory.append(
            (cur_state, action, reward, next_state)
            )

    def learn(self):
        if self.learn_count % self.upgrade_net_iter == 0:
            self.session.run(self.upgrade_network)
            print "upgrade_network."
        # random choice sample
        sample_cnt = self.batch_size if self.batch_size < len(self.memory) else len(self.memory)
        idx = random.sample(range(len(self.memory)), sample_cnt)
        cur_state = []
        action = []
        reward = []
        next_state = []
        for i in idx:
            cur_state.append(self.memory[i][0])
            action.append(self.memory[i][1])
            reward.append(self.memory[i][2])
            next_state.append(self.memory[i][3])
        _, loss = self.session.run([self.train, self.loss]
            , feed_dict = {
                self.cur_state : cur_state
                , self.action : action
                , self.reward : reward
                , self.next_state : next_state
            })
        self.history_loss.append(loss)
        #increate epsilon
        self.epsilon = self.epsilon + self.epsilon_increasement
        if self.epsilon > self.max_epsilon:
            self.epsilon = self.max_epsilon
        self.learn_count += 1
        return loss

    def build_net(self):
        ## input
        self.cur_state = tf.placeholder(tf.float32, [None, self.feature_cnt], name = "cur_state")
        self.next_state = tf.placeholder(tf.float32, [None, self.feature_cnt], name = "next_state")
        self.reward = tf.placeholder(tf.float32, [None, ], name = "reward")
        self.action = tf.placeholder(tf.int32, [None, ], name = "action")
        
        w_initializer = tf.random_normal_initializer(0, 0.3)
        b_initializer = tf.constant_initializer(0.1)
        ## evaluate net
        with tf.variable_scope('evaluate_net'):
            eval_l1 = tf.layers.dense(self.cur_state, 64, tf.nn.relu
                , kernel_initializer = w_initializer
                , bias_initializer = b_initializer
                , name = "eval_l1")
            eval_l2 = tf.layers.dense(eval_l1, 16, tf.nn.relu
                , kernel_initializer = w_initializer
                , bias_initializer = b_initializer
                , name = "eval_l2")
            self.q_eval = tf.layers.dense(eval_l2, self.action_cnt
                , kernel_initializer = w_initializer
                , bias_initializer = b_initializer
                , name = "q_eval")
        ## target net
        with tf.variable_scope("target_net"):
            target_l1 = tf.layers.dense(self.next_state, 64, tf.nn.relu
                , kernel_initializer = w_initializer
                , bias_initializer = b_initializer
                , name = "target_l1")
            target_l2 = tf.layers.dense(target_l1, 16, tf.nn.relu
                , kernel_initializer = w_initializer
                , bias_initializer = b_initializer
                , name = "target_l2")
            self.q_target = tf.layers.dense(target_l2, self.action_cnt
                , kernel_initializer = w_initializer
                , bias_initializer = b_initializer
                , name = "q_target")
        ## 用target_net 和 输入的states, actions , rewards 计算实际的q值
        with tf.variable_scope("calc_real_q"):
            q_real = self.reward + self.reward_decent_rate * tf.reduce_max(
                    self.q_target, axis=1, name = "maxQ_in_qtarget"
                )
            # 冻结q_real依赖的参数在训练过程中被修改
            self.q_real = tf.stop_gradient(q_real)
        ## 用evaluate net预测接下来的步骤
        with tf.variable_scope("predict_action"):
            """
            a_indices = [
                [0, action[0]].   第0组数据， 实际选择的action 标识数字
                [1, action[1]].   第1组数据， 实际选择的action
                ...
                [len(actions] - 1, action[-1])]
            ]
            """
            a_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype = tf.int32), self.action]
                , axis = 1
                )
            """
            q_predict = [
                eval for 0        第0组的eval值
                , eval for 1
                ...
                eval for len(action) -1
            ]
            """
            self.q_predict = tf.gather_nd(params = self.q_eval, indices = a_indices, name = "q_predict")
        ## loss
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_real, self.q_predict))
        ## train eval net
        with tf.variable_scope("train"):
            self.train = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        ## 更新网络
        evaluate_net_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="evaluate_net")
        target_net_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        with tf.variable_scope("upgrade_network"):
            self.upgrade_network = [
                # target_net_weights = evaluate_net_weights
                tf.assign(t, e) for t, e in zip(target_net_weights, evaluate_net_weights)
            ]


    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.history_loss)), self.history_loss)
        plt.ylabel("loss")
        plt.xlabel("train step")
        plt.show()

    def close(self):
        self.session.close()


def play(env, dqn):
    env.reset_env()
    is_terminate = False
    while not is_terminate:
        s = env.get_map_state()
        act_idx = dqn.choose_action(s)
        action = env.get_all_actions()[act_idx]
        print str(env)
        print "action: " + action
        s, is_terminate , _ = env.do_action(action)
        time.sleep(1)


if __name__ == '__main__':
    set_rand_seed(7)
    # 不好搞定
    #env = MazeEnv(20, 30, block_rate = 0.25)
    #env = MazeEnv(19, 19, block_rate = 0.25)
    env = MazeEnv(9, 9, block_rate = 0.35)
    all_actions = env.get_all_actions()
    dqn = DeepQNetwork(
        feature_cnt = env.row_count * env.col_count
        , action_cnt = len(all_actions)
        , memory_size = 10240
        , upgrade_net_iter = 32
        , batch_size = 1024
        , epsilon_increasement = 0.01
        )
    # 可视化
    logDir = "logs/dqn"
    # summaries合并
    merged = tf.summary.merge_all()
    trainWriter = tf.summary.FileWriter(logDir + '/train', dqn.session.graph)
    for epoch in range(1000):
        env.reset_env()
        cur_state = env.get_map_state()
        is_terminate = False
        data = []
        while not is_terminate:
            act_idx = dqn.choose_action(cur_state)
            action = all_actions[act_idx]
            #print "action: " + str(action)
            (next_state, is_terminate, reward) = env.do_action(action)
            next_state = env.get_map_state()
            data.append((cur_state, act_idx, reward, next_state))
            cur_state = next_state
        for i in range(len(data) - 1, -1, -1):
            cur_state, action, r, next_state= data[i]
            """ reward / float(len(data) - i) """
            dqn.store_train(cur_state, action, r , next_state)
        dqn.store_train
        loss = dqn.learn()
        print "epoch %d loss %0.4lf reward %d" % (epoch, loss, reward)
    play(env, dqn)
    dqn.plot_loss()
    dqn.close()



