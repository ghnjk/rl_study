#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import random
import memory


def set_rand_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


class DDPG(object):

    def __init__(self, fearture_count, action_count, action_bound,
                 batch_size=32,
                 memory_size=10000,
                 actor_learning_rate=0.001,
                 critic_learning_rate=0.002,
                 soft_replacement=0.01,
                 reward_decay=0.9):
        self.feature_count = fearture_count
        self.action_count = action_count
        self.action_bound = action_bound  # 行为区间的最大值
        self.batch_size = batch_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.soft_replacement = soft_replacement
        self.reward_decay = reward_decay

        self.memory = memory.RandomMemory(memory_size=memory_size)

        # tf inputs
        self.tf_cur_state = None
        self.tf_next_state = None
        self.tf_reward = None
        # tf network variables
        self.tf_eval_actor_vars = None
        self.tf_target_actor_vars = None
        self.tf_eval_critic_vars = None
        self.tf_target_critic_vars = None
        # tf outputs
        self.tf_q_eval = None
        self.tf_q_target = None
        self.tf_a_eval = None
        self.tf_a_target = None
        self.tf_td_error = None
        # tf ops
        self.tf_soft_replace_vars_op = None
        self.tf_actor_train_op = None
        self.tf_critic_train_op = None

        self.build_net()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        # inputs
        self.tf_cur_state = tf.placeholder(tf.float32, [None, self.feature_count],  name="cur_state")
        self.tf_next_state = tf.placeholder(tf.float32, [None, self.feature_count], name="next_state")
        self.tf_reward = tf.placeholder(tf.float32, [None, 1], name="reward")

        with tf.variable_scope("actor"):
            self.tf_a_eval = self._build_actor_net(self.tf_cur_state, scope="eval_net", trainable=True)
            self.tf_a_target = self._build_actor_net(self.tf_next_state, scope="target_net", trainable=False)

        with tf.variable_scope("critic"):
            self.tf_q_eval = self._build_critic_net(self.tf_cur_state,
                                                    self.tf_a_eval,
                                                    scope="eval_net",
                                                    trainable=True)
            self.tf_q_target = self._build_critic_net(self.tf_next_state,
                                                      self.tf_a_target,
                                                      scope="target_net",
                                                      trainable=False)

        # get network variables
        self.tf_eval_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="actor/eval_net")
        self.tf_target_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="actor/target_net")
        self.tf_eval_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="critic/eval_net")
        self.tf_target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="critic/target_net")

        with tf.variable_scope("soft_update_network"):
            self.tf_soft_replace_vars_op = [
                [
                    tf.assign(
                        ta, (1 - self.soft_replacement) * ta + self.soft_replacement * ea
                    ),
                    tf.assign(
                        tc, (1 - self.soft_replacement) * tc + self.soft_replacement * ec
                    )
                ]
                for ta, ea, tc, ec in zip(self.tf_target_actor_vars,
                                          self.tf_eval_actor_vars,
                                          self.tf_target_critic_vars,
                                          self.tf_eval_critic_vars)
            ]

        with tf.variable_scope("q_target"):
            q_target = self.tf_reward + self.reward_decay * self.tf_q_target
        with tf.variable_scope("td_error"):
            self.tf_td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.tf_q_eval)
        self.tf_critic_train_op = tf.train.AdamOptimizer(self.critic_learning_rate)\
            .minimize(self.tf_td_error,
                      var_list=self.tf_eval_critic_vars)

        #  because q_eval is calc by cur_state and act_eval
        #  trainning act_eval is to maximized the q
        with tf.variable_scope("act_loss"):
            act_loss = -tf.reduce_mean(self.tf_q_eval)
        self.tf_actor_train_op = tf.train.AdamOptimizer(self.actor_learning_rate)\
            .minimize(act_loss,
                      var_list=self.tf_eval_actor_vars)

    def choose_action(self, cur_state):
        cur_state = np.array(cur_state)[np.newaxis, :]
        return self.sess.run(self.tf_a_eval, feed_dict={
            self.tf_cur_state: cur_state
        })[0]

    def learn(self):
        self.sess.run(self.tf_soft_replace_vars_op)

        samples = self.memory.choose_sample(sample_count=self.batch_size)
        cur_state = []
        action = []
        reward = []
        next_state = []
        for item in samples:
            cur_state.append(item[0])
            action.append(item[1])
            reward.append(item[2])
            next_state.append(item[3])

        self.sess.run(self.tf_actor_train_op, feed_dict={
            self.tf_cur_state: cur_state
        })
        self.sess.run(self.tf_critic_train_op, feed_dict={
            self.tf_cur_state: cur_state,
            self.tf_a_eval: action,
            self.tf_reward: reward,
            self.tf_next_state: next_state
        })

    def store(self, cur_state, action, reward, next_state):
        reward = np.array(reward).reshape(1)
        item = [cur_state, action, reward, next_state]
        self.memory.append(item)

    def _build_actor_net(self, state, scope, trainable):
        with tf.variable_scope(scope):
            layer_1 = tf.layers.dense(inputs=state,
                                      units=32,
                                      activation=tf.nn.relu,
                                      name="layer_1",
                                      trainable=trainable)
            act = tf.layers.dense(inputs=layer_1,
                                  units=self.action_count,
                                  activation=tf.nn.tanh,
                                  name="act_probs",
                                  trainable=trainable)
            return tf.multiply(act, self.action_bound, name="scale_act")

    def _build_critic_net(self, state, act, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 32
            w1_s = tf.get_variable("w1_s", [self.feature_count, n_l1], trainable=trainable)
            w1_a = tf.get_variable("w1_a", [self.action_count, n_l1], trainable=trainable)
            bias = tf.get_variable("bias", [1, n_l1], trainable=trainable)
            layer = tf.nn.relu(tf.matmul(state, w1_s) + tf.matmul(act, w1_a) + bias)
            return tf.layers.dense(inputs=layer,
                                   units=1,
                                   trainable=trainable,
                                   name="q_value")


def run_pendulum():
    import gym

    set_rand_seed(7)

    env = gym.make("Pendulum-v0")
    env = env.unwrapped
    env.seed(1)
    max_step_per_epoch = 200
    max_epoch = 200

    feature_count = env.observation_space.shape[0]
    action_count = env.action_space.shape[0]
    action_bound = env.action_space.high

    ddpg = DDPG(feature_count, action_count, action_bound)

    log_dir = "logs/ddpg"
    train_writer = tf.summary.FileWriter(log_dir + '/train', ddpg.sess.graph)
    need_render = False
    act_random_var = 3
    for epoch in range(max_epoch):
        cur_state = env.reset()
        ep_reward = 0
        for j in range(max_step_per_epoch):
            if need_render:
                env.render()
            action = ddpg.choose_action(cur_state)
            # add randomness to action selection for exploration
            action = np.clip(np.random.normal(action, act_random_var), -2, 2)
            next_state, reward, is_done, info = env.step(action)
            ddpg.store(cur_state, action, reward / 10, next_state)
            if ddpg.memory.data_count() >= ddpg.memory.memory_size:
                # decay the action randomness
                act_random_var *= 0.9995
                ddpg.learn()
            cur_state = next_state
            ep_reward += reward
            if j == max_step_per_epoch - 1:
                print("epoch ", epoch, " Reward: ", int(ep_reward), " explors: ", round(act_random_var, 2))
                if ep_reward > -500:
                    need_render = True
    train_writer.close()


if __name__ == '__main__':
    run_pendulum()
