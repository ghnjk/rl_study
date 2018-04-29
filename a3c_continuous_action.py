#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import random

global_running_r = []
global_running_ep = 0
max_global_ep = 2000
max_step_per_epoch = 200
global_coord = None
update_global_iter = 10
GAME_NAME = "Pendulum-v0"


def set_rand_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


class ACNet(object):
    def __init__(self, feature_count, action_count, action_bound,
                 sess,
                 scope, global_acnet=None,
                 entropy_beta=0.01,
                 actor_optimizer=None,
                 critic_optimizer=None):
        self.feature_count = feature_count
        self.action_count = action_count
        self.action_bound = action_bound
        self.global_acnet = global_acnet
        self.entropy_beta = entropy_beta
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.sess = sess

        # tf inputs
        self.tf_cur_state = None
        self.tf_action_history = None
        self.tf_target_v = None
        # tf network variables
        self.tf_actor_variables = None
        self.tf_critic_variables = None
        # tf outputs
        self.tf_eval_v = None
        self.tf_critic_loss = None
        self.tf_entropy_v = None
        self.tf_actor_loss = None
        self.tf_actor_gradient = None
        self.tf_critic_gradient = None
        self.tf_eval_action = None
        # tf op
        self.tf_pull_actor_variable_op = None
        self.tf_pull_critic_variable_op = None
        self.tf_push_actor_gradient_op = None
        self.tf_push_critic_gradient_op = None

        if global_acnet is None:
            # build current as global net
            with tf.variable_scope(scope):
                self._build_acnet(scope)
        else:
            # build current as worker net
            with tf.variable_scope(scope):
                self._build_worker_net(scope)

    def _build_worker_net(self, scope):
        self.tf_action_history = tf.placeholder(tf.float32, [None, self.action_count], name="action_history")
        self.tf_target_v = tf.placeholder(tf.float32, [None, 1], name="target_v")
        mu, sigma, _ = self._build_acnet(scope)
        td_error = tf.subtract(self.tf_target_v, self.tf_eval_v, name="td_error")
        with tf.name_scope("critic_loss"):
            self.tf_critic_loss = tf.reduce_mean(tf.square(td_error))
        with tf.name_scope("wrap_a_out"):
            mu, sigma = mu * self.action_bound[1], sigma + 1e-4
        normal_distribute = tf.distributions.Normal(mu, sigma)
        with tf.name_scope("actor_loss"):
            log_prob = normal_distribute.log_prob(self.tf_action_history)
            exp_v = log_prob * tf.stop_gradient(td_error)
            entropy = normal_distribute.entropy()
            self.tf_entropy_v = self.entropy_beta * entropy + exp_v
            self.tf_actor_loss = tf.reduce_mean(-self.tf_entropy_v)
        with tf.name_scope("choose_action"):
            self.tf_eval_action = tf.clip_by_value(tf.squeeze(normal_distribute.sample(1), axis=0),
                                                   self.action_bound[0], self.action_bound[1])
        with tf.name_scope("local_gradient"):
            self.tf_actor_gradient = tf.gradients(self.tf_actor_loss, self.tf_actor_variables)
            self.tf_critic_gradient = tf.gradients(self.tf_critic_loss, self.tf_critic_variables)
        with tf.name_scope("sync"):
            with tf.name_scope("pull"):
                self.tf_pull_actor_variable_op = [
                    lp.assign(gp) for lp, gp in zip(self.tf_actor_variables,
                                                    self.global_acnet.tf_actor_variables)
                ]
                self.tf_pull_critic_variable_op = [
                    lp.assign(gp) for lp, gp in zip(self.tf_critic_variables,
                                                    self.global_acnet.tf_critic_variables)
                ]
            with tf.name_scope("push"):
                self.tf_push_actor_gradient_op = self.actor_optimizer.apply_gradients(
                    zip(self.tf_actor_gradient, self.global_acnet.tf_actor_variables)
                )
                self.tf_push_critic_gradient_op = self.critic_optimizer.apply_gradients(
                    zip(self.tf_critic_gradient, self.global_acnet.tf_critic_variables)
                )

    def update_to_global(self, feed_dict):
        self.sess.run([self.tf_push_actor_gradient_op, self.tf_push_critic_gradient_op],
                      feed_dict=feed_dict)

    def pull_from_global(self):
        self.sess.run([self.tf_pull_actor_variable_op, self.tf_pull_critic_variable_op])

    def choose_action(self, cur_state):
        cur_state = np.array(cur_state)[np.newaxis, :]
        return self.sess.run(self.tf_eval_action, feed_dict={
            self.tf_cur_state: cur_state
        })[0]

    def _build_acnet(self, scope):
        self.tf_cur_state = tf.placeholder(tf.float32, [None, self.feature_count], name="cur_state")
        w_init = tf.random_normal_initializer(0, 0.1)
        with tf.variable_scope("actor"):
            layer_a = tf.layers.dense(inputs=self.tf_cur_state,
                                      units=200,
                                      activation=tf.nn.relu6,
                                      kernel_initializer=w_init,
                                      name="layer_a")
            mu = tf.layers.dense(inputs=layer_a,
                                 units=self.action_count,
                                 activation=tf.nn.tanh,
                                 kernel_initializer=w_init,
                                 name="mu")
            sigma = tf.layers.dense(inputs=layer_a,
                                    units=self.action_count,
                                    activation=tf.nn.softplus,
                                    kernel_initializer=w_init,
                                    name="sigma")
        with tf.variable_scope("critic"):
            layer_c = tf.layers.dense(inputs=self.tf_cur_state,
                                      units=100,
                                      activation=tf.nn.relu6,
                                      kernel_initializer=w_init,
                                      name="layer_c")
            self.tf_eval_v = tf.layers.dense(inputs=layer_c,
                                             units=1,
                                             kernel_initializer=w_init,
                                             name="v")
        self.tf_actor_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + "/actor")
        self.tf_critic_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + "/critic")
        return mu, sigma, self.tf_eval_v


class Worker(object):
    def __init__(self, name,
                 feature_count,
                 action_count,
                 action_bound,
                 sess,
                 global_acnet,
                 actor_optimizer,
                 critic_optimizer,
                 reward_decay=0.9):
        import gym
        self.env = gym.make(GAME_NAME).unwrapped
        self.name = name
        self.sess = sess
        self.reward_decay = reward_decay
        self.acnet = ACNet(feature_count=feature_count,
                           action_count=action_count,
                           action_bound=action_bound,
                           sess=sess,
                           scope=self.name,
                           global_acnet=global_acnet,
                           actor_optimizer=actor_optimizer,
                           critic_optimizer=critic_optimizer
                           )

    def work(self):
        global global_running_ep
        global global_running_r
        total_step = 1
        buffer_cur_state = []
        buffer_action = []
        buffer_reward = []
        while not global_coord.should_stop() and global_running_ep < max_global_ep:
            cur_state = self.env.reset()
            exp_r = 0
            for step in range(max_step_per_epoch):
                action = self.acnet.choose_action(cur_state)
                next_state, reward, is_done, info = self.env.step(action)
                is_done = True if step == max_step_per_epoch - 1 else False
                exp_r += reward
                buffer_cur_state.append(cur_state)
                buffer_action.append(action)
                buffer_reward.append((reward + 8) / 8)

                if total_step % update_global_iter == 0 or is_done:
                    if is_done:
                        v_s_ = 0
                    else:
                        v_s_ = self.sess.run(self.acnet.tf_eval_v, feed_dict={
                            self.acnet.tf_cur_state: next_state[np.newaxis, :]
                        })[0, 0]
                    buffer_v_target = []
                    for r in buffer_reward[::-1]:
                        v_s_ = r + self.reward_decay * v_s_
                        buffer_v_target.append([v_s_])
                    buffer_v_target.reverse()
                    buffer_cur_state = np.vstack(buffer_cur_state)
                    buffer_action = np.vstack(buffer_action)
                    feed_dict = {
                        self.acnet.tf_cur_state: buffer_cur_state,
                        self.acnet.tf_action_history: buffer_action,
                        self.acnet.tf_target_v: buffer_v_target
                    }
                    self.acnet.update_to_global(feed_dict)
                    buffer_cur_state = []
                    buffer_action = []
                    buffer_reward = []
                    self.acnet.pull_from_global()

                cur_state = next_state
                total_step += 1
                if is_done:
                    if len(global_running_r) == 0:
                        global_running_r.append(exp_r)
                    else:
                        global_running_r.append(global_running_r[-1] * 0.9 + exp_r * 0.1)
                    print(self.name,
                          "epoch",
                          global_running_ep,
                          "ep_r",
                          int(global_running_r[-1])
                          )
                    global_running_ep += 1
                    break


def run_pendulum():
    import gym
    import threading
    import matplotlib.pyplot as plt

    actor_learning_rate = 0.0001
    critic_learning_rate = 0.001

    env = gym.make(GAME_NAME)
    feature_count = env.observation_space.shape[0]
    action_count = env.action_space.shape[0]
    action_bound = [env.action_space.low, env.action_space.high]

    sess = tf.Session()
    workers = []

    with tf.device("/cpu:0"):
        actor_optimizer = tf.train.RMSPropOptimizer(actor_learning_rate, name="actor_optimizer")
        critic_optimizer = tf.train.RMSPropOptimizer(critic_learning_rate, name="critic_optimizer")
        global_acnet = ACNet(feature_count=feature_count,
                             action_count=action_count,
                             action_bound=action_bound,
                             sess=sess,
                             scope="global_ac")
        for i in range(4):
            worker_name = "worker_%d" % i
            w = Worker(
                name=worker_name,
                feature_count=feature_count,
                action_count=action_count,
                action_bound=action_bound,
                sess=sess,
                global_acnet=global_acnet,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer
            )
            workers.append(w)

    global global_coord
    global_coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    log_dir = "logs/a3c_continuous"
    tf.summary.FileWriter(log_dir, sess.graph)
    worker_threads = []
    for worker in workers:
        t = threading.Thread(target=lambda: worker.work())
        t.start()
        worker_threads.append(t)
    global_coord.join(worker_threads)

    plt.plot(np.arange(len(global_running_r)), global_running_r)
    plt.xlabel("step")
    plt.ylabel("total moving reword")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    run_pendulum()
