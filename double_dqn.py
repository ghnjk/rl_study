#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
from maze import *
from collections import deque
import random


def set_rand_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


class DoubleDQN(object):
    def __init__(self, feature_cnt, action_cnt,
                 upgrade_net_iter,
                 batch_size=32,
                 memory_size=2000,
                 reward_decent_rate=0.9,
                 learning_rate=0.01,
                 epsilon_increasement=None,
                 max_epsilon=0.9,
                 is_double_dqn=True):
        """
        feature_cnt: 输入特征总数
        action_cnt: 输出选择的动作总数
        upgrade_net_iter： 更新target net的频率
        batch_size: 每次选择的样本数据大小
        memory_size: 所有样本记录大小
        """
        self.feature_cnt = feature_cnt
        self.action_cnt = action_cnt
        self.upgrade_net_iter = upgrade_net_iter
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.reward_decent_rate = reward_decent_rate
        self.learn_rate = learning_rate
        self.epsilon_increasement = epsilon_increasement
        self.max_epsilon = max_epsilon
        self.is_double_dqn = is_double_dqn
        if self.epsilon_increasement is None:
            self.epsilon = self.max_epsilon
        else:
            self.epsilon = 0
        # 临时
        self.memory = deque(maxlen=self.memory_size)
        self.learn_count = 0
        self.running_q = 0

        # network inputs
        self.cur_state_in = None
        self.next_state_in = None
        self.reward_in = None
        self.action_in = None
        self.q_target_in = None
        # network outputs
        self.q_eval = None
        self.q_next = None
        self.loss = None
        self.train = None
        self.upgrade_network = None

        # 建立网络
        self.build_net()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.history_loss = []
        self.history_q = []

    def build_net(self):
        # input
        self.cur_state_in = tf.placeholder(tf.float32, [None, self.feature_cnt], name="cur_state")
        self.next_state_in = tf.placeholder(tf.float32, [None, self.feature_cnt], name="next_state")
        self.reward_in = tf.placeholder(tf.float32, [None, ], name="reward")
        self.action_in = tf.placeholder(tf.int32, [None, ], name="action")
        self.q_target_in = tf.placeholder(tf.float32, [None, self.action_cnt], name="q_target")

        w_initializer = tf.random_normal_initializer(0, 0.3)
        b_initializer = tf.constant_initializer(0.1)
        # evaluate net
        with tf.variable_scope('evaluate_net'):
            eval_l1 = tf.layers.dense(self.cur_state_in, 32, tf.nn.relu,
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer,
                                      name="eval_l1")
            self.q_eval = tf.layers.dense(eval_l1, self.action_cnt,
                                          kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer,
                                          name="q_eval")
        # target net
        with tf.variable_scope("target_net"):
            target_l1 = tf.layers.dense(self.next_state_in, 32, tf.nn.relu,
                                        kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer,
                                        name="target_net_l1")
            self.q_next = tf.layers.dense(target_l1, self.action_cnt,
                                          kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer,
                                          name="q_next")
        # loss
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target_in, self.q_eval))
        # train eval net
        with tf.variable_scope("train"):
            self.train = tf.train.RMSPropOptimizer(self.learn_rate).minimize(self.loss)
        # 更新网络
        evaluate_net_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="evaluate_net")
        target_net_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        with tf.variable_scope("upgrade_network"):
            self.upgrade_network = [
                # target_net_weights = evaluate_net_weights
                tf.assign(t, e) for t, e in zip(target_net_weights, evaluate_net_weights)
            ]

    def choice_action(self, observation):
        action_values = self.session.run(self.q_eval,
                                         feed_dict={
                                             self.cur_state_in: [observation]
                                         })
        act = np.argmax(action_values)
        self.running_q = self.running_q * 0.99 + 0.01 * np.max(action_values)
        self.history_q.append(self.running_q)
        if np.random.uniform() > self.epsilon:
            act = np.random.randint(0, self.action_cnt)
        return act

    def learn(self, show_msg=True):
        if self.learn_count % self.upgrade_net_iter == 0:
            self.session.run(self.upgrade_network)
            if show_msg:
                print "upgrade_network."
        # random choice sample
        sample_cnt = self.batch_size if self.batch_size < len(self.memory) else len(self.memory)
        idx = random.sample(range(len(self.memory)), sample_cnt)
        c_s = []
        act = []
        r = []
        n_s = []
        for i in idx:
            c_s.append(self.memory[i][0])
            act.append(self.memory[i][1])
            r.append(self.memory[i][2])
            n_s.append(self.memory[i][3])
        # train
        target4next, eval4next = self.session.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.cur_state_in: n_s,
                self.next_state_in: n_s
            }
        )
        eval4cur = self.session.run(
            self.q_eval,
            feed_dict={
                self.cur_state_in: c_s
            }
        )
        # 用老的target_net选择下一个节点的q
        batch_index = np.arange(sample_cnt, dtype=np.int32)
        # print batch_index
        if self.is_double_dqn:
            max_eval4next = np.argmax(eval4next, axis=1)
            # print max_eval4next
            selected_q_next = target4next[batch_index, max_eval4next]
        else:
            selected_q_next = np.max(target4next, axis=1)
        # 更新q_target
        q_target = eval4cur.copy()
        q_target[batch_index, act] = r + self.reward_decent_rate * selected_q_next
        # 训练evalute_net参数
        _, cost = self.session.run(
            [self.train, self.loss],
            feed_dict={
                self.cur_state_in: c_s,
                self.q_target_in: q_target
            }
        )
        self.history_loss.append(cost)
        # increate epsilon
        self.epsilon = self.epsilon + self.epsilon_increasement
        if self.epsilon > self.max_epsilon:
            self.epsilon = self.max_epsilon
        self.learn_count += 1
        return cost

    def store_train(self, cur_state, action, reward, next_state):
        self.memory.append(
            (cur_state, action, reward, next_state)
        )

    def close(self):
        self.session.close()

    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.history_loss)), self.history_loss)
        plt.ylabel("loss")
        plt.xlabel("train step")
        plt.show()

    def plot_history_q(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.history_q)), self.history_q)
        plt.xlabel("train step")
        plt.ylabel("history q")
        plt.grid()
        plt.show()


def play(env, dqn):
    import time
    env.reset_env()
    is_terminate = False
    while not is_terminate:
        # s = env.get_map_state()
        s = [env.get_state()]
        act_idx = dqn.choice_action(s)
        action = env.get_all_actions()[act_idx]
        print str(env)
        print "action: " + action
        s, is_terminate, _ = env.do_action(action)
        time.sleep(1)


def train_maze():
    set_rand_seed(7)
    # 不好搞定
    # env = MazeEnv(20, 30, block_rate = 0.25)
    # env = MazeEnv(19, 19, block_rate=0.25)
    env = MazeEnv(9, 9, block_rate=0.1)
    all_actions = env.get_all_actions()
    dqn = DoubleDQN(
        feature_cnt=1,
        action_cnt=len(all_actions),
        memory_size=3000,
        upgrade_net_iter=50,
        batch_size=32,
        epsilon_increasement=0.001
        # is_double_dqn=False
    )
    step_count = 0
    loss = 0
    reward = 0
    for epoch in range(10000):
        env.reset_env()
        cur_state = [env.get_state()]
        is_terminate = False
        data = []
        while not is_terminate:
            act_idx = dqn.choice_action(cur_state)
            action = all_actions[act_idx]
            (next_state, is_terminate, reward) = env.do_action(action)
            step_count += 1
            next_state = [env.get_state()]
            dqn.store_train(cur_state, act_idx, reward, next_state)
            if step_count > dqn.memory_size:
                loss = dqn.learn(False)
            data.append((cur_state, act_idx, reward, next_state))
            cur_state = next_state
        if step_count > dqn.memory_size and epoch % 100 == 0:
            print "epoch %d loss %0.4lf reward %d last_q: %0.4lf" % (epoch, loss, reward, dqn.history_q[-1])
    play(env, dqn)
    # dqn.plot_loss()
    dqn.plot_history_q()
    dqn.close()


if __name__ == '__main__':
    train_maze()
