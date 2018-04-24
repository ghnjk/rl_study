#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import random


def set_rand_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


class PolicyGradient(object):

    def __init__(self, action_count, feature_count, learning_rate=0.01, reward_decay=0.95):
        self.action_count = action_count
        self.feature_count = feature_count
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay

        self.data_obs = []
        self.data_act = []
        self.data_r = []

        # network inputs
        self.tf_observation = None
        self.tf_action = None
        self.tf_vt = None
        # network outputs
        self.tf_all_act_probs = None
        self.tf_loss = None
        self.train_op = None

        self.build_net()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def build_net(self):
        with tf.name_scope("inputs"):
            self.tf_observation = tf.placeholder(tf.float32, [None, self.feature_count], name="tf_observation")
            self.tf_action = tf.placeholder(tf.int32, [None, ], name="tf_action")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="tf_vt")
        fc1 = tf.layers.dense(self.tf_observation, units=10,
                              activation=tf.nn.tanh,
                              kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                              bias_initializer=tf.constant_initializer(0.1),
                              name="fc1")
        all_acts = tf.layers.dense(fc1, units=self.action_count,
                                   activation=None,
                                   kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                   bias_initializer=tf.constant_initializer(0.1),
                                   name="fc2")
        self.tf_all_act_probs = tf.nn.softmax(all_acts, name="all_act_probs")

        with tf.name_scope("loss"):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_acts, labels=self.tf_action)
            # or this way
            neg_log_prob = tf.reduce_sum(-tf.log(self.tf_all_act_probs) * tf.one_hot(self.tf_action, self.action_count),
                                         axis=1)
            self.tf_loss = tf.reduce_mean(neg_log_prob * self.tf_vt)

        with tf.name_scope("train"):
            self.train_op = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.tf_loss)

    def choose_action(self, observation):
        act_probs = self.session.run(self.tf_all_act_probs,
                                     feed_dict={
                                         self.tf_observation: observation[np.newaxis, :]
                                     })
        acts = np.random.choice(range(act_probs.shape[1]), p=act_probs.ravel())
        return acts

    def store_transition(self, observation, actions, reward):
        self.data_obs.append(observation)
        self.data_act.append(actions)
        self.data_r.append(reward)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        self.session.run(self.train_op, feed_dict={
            self.tf_observation: np.vstack(self.data_obs),
            self.tf_action: np.array(self.data_act),
            self.tf_vt: discounted_ep_rs_norm
        })
        # clean memory buffer
        self.data_obs = []
        self.data_act = []
        self.data_r = []
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.data_r)
        running_add = 0
        for idx in reversed(range(0, len(self.data_r))):
            running_add = running_add * self.reward_decay + self.data_r[idx]
            discounted_ep_rs[idx] = running_add
        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs


def run_mountain_car():
    import gym
    set_rand_seed(7)
    env = gym.make('MountainCar-v0')
    env.seed(1)
    env = env.unwrapped
    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    rl = PolicyGradient(action_count=env.action_space.n,
                        feature_count=env.observation_space.shape[0],
                        learning_rate=0.02,
                        reward_decay=0.995,)

    running_reward = None

    for epoch in range(1000):
        observation = env.reset()
        while True:
            if running_reward != None and running_reward >= -2000:
                env.render()
            action = rl.choose_action(observation)
            next_observation, reward, is_done, info = env.step(action)
            rl.store_transition(observation, action, reward)
            if is_done:
                reward_sum = sum(rl.data_r)
                if running_reward is None:
                    running_reward = reward_sum
                else:
                    running_reward = running_reward * 0.99 + reward_sum * 0.01
                print("epoch", epoch, " step_count: ", len(rl.data_act), " reward: ", int(running_reward))
                vt = rl.learn()
                if epoch == 10:
                    import matplotlib.pyplot as plt
                    plt.plot(vt)
                    plt.xlabel("steps")
                    plt.ylabel("normalized state-action value")
                    plt.show()
                break
            observation = next_observation


if __name__ == '__main__':
    run_mountain_car()
