#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import random


def set_rand_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


class Actor(object):

    def __init__(self, sess, feature_count, action_count, learning_rate=0.001):
        self.sess = sess
        self.feature_count = feature_count
        self.action_count = action_count
        self.learning_rate = learning_rate

        # tf inputs
        self.tf_cur_state = None
        self.tf_td_error = None
        self.tf_action = None
        # tf outputs
        self.tf_all_act_probs = None
        self.tf_exp_v = None
        self.tf_train_op = None
        self.tf_loss = None

        self.build_net()

    def build_net(self):
        with tf.name_scope("inputs"):
            self.tf_cur_state = tf.placeholder(tf.float32, [1, self.feature_count], name="cur_state")
            self.tf_td_error = tf.placeholder(tf.float32, None, name="td_error")
            self.tf_action = tf.placeholder(tf.int32, None, name="action")
        with tf.variable_scope("actor"):
            layer_1 = tf.layers.dense(inputs=self.tf_cur_state, units=20,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(0, 1),
                                      bias_initializer=tf.constant_initializer(0.1),
                                      name="layer_1")
            self.tf_all_act_probs = tf.layers.dense(inputs=layer_1,
                                                    units=self.action_count,
                                                    activation=tf.nn.softmax,
                                                    kernel_initializer=tf.random_normal_initializer(0, 1),
                                                    bias_initializer=tf.constant_initializer(0.1),
                                                    name="all_act_probs")
        with tf.name_scope("exp_v"):
            log_prob = tf.log(self.tf_all_act_probs[0, self.tf_action])
            self.tf_exp_v = tf.reduce_mean(log_prob * self.tf_td_error)
        with tf.name_scope("train"):
            self.tf_train_op = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(-self.tf_exp_v)

    def learn(self, cur_state, action, td_error):
        cur_state = np.array(cur_state)[np.newaxis, :]
        feed_dict = {
            self.tf_cur_state: cur_state,
            self.tf_td_error: td_error,
            self.tf_action: action
        }
        _, exp_v = self.sess.run([self.tf_train_op, self.tf_exp_v], feed_dict=feed_dict)
        return exp_v

    def choose_action(self, cur_state):
        cur_state = np.array(cur_state).reshape(1, 1)
        all_probs = self.sess.run(self.tf_all_act_probs,
                                  feed_dict={self.tf_cur_state: cur_state})
        return np.random.choice(np.arange(all_probs.shape[1]), p=all_probs.ravel())


class Critic(object):

    def __init__(self, sess, feature_count, learning_rate=0.01, reward_decay=0.9):
        self.sess = sess
        self.feature_count = feature_count
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay

        # tensor flow input
        self.tf_cur_state = None
        self.tf_next_state_v = None
        self.tf_reward = None
        # tensor flow output
        self.tf_predict_v = None
        self.tf_td_error = None
        self.tf_loss = None
        self.tf_train_op = None

        self.build_net()

    def build_net(self):
        self.tf_cur_state = tf.placeholder(tf.float32, [1, self.feature_count], name="cur_state")
        self.tf_next_state_v = tf.placeholder(tf.float32, [1, 1], name="next_state_v")
        self.tf_reward = tf.placeholder(tf.float32, None, name="reward")
        with tf.variable_scope("critic"):
            layer_1 = tf.layers.dense(inputs=self.tf_cur_state,
                                      units=20,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(0, 1),
                                      bias_initializer=tf.constant_initializer(0.1),
                                      name="layer_1")
            self.tf_predict_v = tf.layers.dense(inputs=layer_1,
                                                units=1,
                                                activation=None,
                                                kernel_initializer=tf.random_normal_initializer(0, 1),
                                                bias_initializer=tf.constant_initializer(0.1),
                                                name="predict_v")
        with tf.name_scope("squre_td_error"):
            self.tf_td_error = self.tf_reward + self.reward_decay * self.tf_next_state_v - self.tf_predict_v
            self.tf_loss = tf.square(self.tf_td_error)
        with tf.name_scope("train"):
            self.tf_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.tf_loss)

    def learn(self, cur_state, reward, next_state):
        cur_state = np.array(cur_state)[np.newaxis, :]
        next_state = np.array(next_state)[np.newaxis, :]
        next_predict_v = self.sess.run(self.tf_predict_v,
                                       feed_dict={self.tf_cur_state: next_state})
        td_error, _ = self.sess.run([self.tf_td_error, self.tf_train_op],
                                    feed_dict={
                                        self.tf_cur_state: cur_state,
                                        self.tf_next_state_v: next_predict_v,
                                        self.tf_reward: reward
                                    })
        return td_error, next_predict_v


def play(env, actor):
    import time
    env.reset_env()
    is_terminate = False
    while not is_terminate:
        # s = env.get_map_state()
        s = [env.get_state()]
        act_idx = actor.choose_action(s)
        action = env.get_all_actions()[act_idx]
        print str(env)
        print "action: " + action
        s, is_terminate, _ = env.do_action(action)
        time.sleep(1)


def train_maze():
    from maze import MazeEnv
    set_rand_seed(7)
    # 不好搞定
    # env = MazeEnv(20, 30, block_rate = 0.25)
    # env = MazeEnv(19, 19, block_rate=0.25)
    env = MazeEnv(9, 9, block_rate=0.1)
    all_actions = env.get_all_actions()
    sess = tf.Session()
    actor = Actor(
        sess=sess,
        feature_count=1,
        action_count=len(all_actions)
    )
    critic = Critic(
        sess=sess,
        feature_count=1
    )
    sess.run(tf.global_variables_initializer())
    log_dir = "logs/actor_critic"
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    step_count = 0
    reward = 0
    for epoch in range(10000):
        env.reset_env()
        cur_state = [env.get_state()]
        is_terminate = False
        while not is_terminate:
            act_idx = actor.choose_action(cur_state=cur_state)
            action = all_actions[act_idx]
            (next_state, is_terminate, reward) = env.do_action(action)
            step_count += 1
            next_state = [env.get_state()]
            td_error, next_predict_v = critic.learn(cur_state=cur_state, reward=reward, next_state=next_state)
            actor.learn(cur_state=cur_state, action=act_idx, td_error=td_error)
            cur_state = next_state
        if epoch % 100 == 0:
            print "epoch %d reward %d" % (epoch, reward)
    play(env, actor)
    train_writer.close()


if __name__ == '__main__':
    train_maze()
