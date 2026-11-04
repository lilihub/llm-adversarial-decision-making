# from https://github.com/keon/deep-q-learning/blob/master/ddqn.py


import random
import numpy as np
import tensorflow.keras.backend as K

import tensorflow as tf
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.ops.init_ops_v2 import VarianceScaling

from arrq import ArrQ
from util import DLogger
from util.logger import LogFile


class DQNAgent:
    def __init__(self, state_size, action_size,
                 buffer_size,
                 epsilon,
                 lr
                 ):
        q_size = buffer_size
        self.state_size = state_size
        self.action_size = action_size
        self.memory = {'state': ArrQ(q_size, [state_size]),
                       'action': ArrQ(q_size, [action_size]),
                       'reward': ArrQ(q_size, [1]),
                       'next_state': ArrQ(q_size, [state_size]),
                       'done': ArrQ(q_size, [1])
                       }
        self.gamma = 1  # discount rate
        self.epsilon = epsilon
        self.learning_rate = lr
        # self.epsilon_min = 0.1
        # self.epsilon_decay = 0.99
        # self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.actions = np.eye(self.action_size)
        self.update_target_model()
        self.next_max = 0
        self.optimizer = Adam(lr=self.learning_rate)

    """Huber loss for Q Learning

    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """
    def _huber_loss(self, y_true, y_pred):
        error = y_true - y_pred
        squared_loss = 0.5 * K.square(error)
        return K.mean(squared_loss)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_shape=(self.state_size,), activation='relu', kernel_initializer=VarianceScaling(scale=2)))
        model.add(Dense(128, activation='relu', kernel_initializer=VarianceScaling(scale=2)))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer=VarianceScaling(scale=2)))
        # model = load_model('../nongit/archive/gonogo/state-reg/RL_gonogo_dqn_vec_bin_temp/RL_nc_dqn_buf_100000_eps_0.2_lr_0.001_lof_1/model-12000.h5',
        #                    custom_objects = {'_huber_loss': self._huber_loss}
        #                    )
        model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory['state'].append(state)
        self.memory['action'].append(action)
        self.memory['reward'].append(np.array(reward[:, np.newaxis]))
        self.memory['next_state'].append(next_state)
        self.memory['done'].append(np.array(done[:, np.newaxis]))

    def act(self, state, eps):
        adv_action = np.argmax(self.model(state), axis=1)
        r_size = int(adv_action.shape[0] * eps)
        adv_action[np.random.choice(np.arange(0, adv_action.shape[0]), r_size)] = np.random.randint(0, self.action_size, r_size)
        return adv_action

        # if np.random.rand() <= self.epsilon:
        #     return np.random.randint(0, self.action_size, state.shape[0])
        # return K.argmax(self.model(state), axis=1).numpy()

    def replay(self, batch_size):
        with tf.GradientTape() as tape:
            batch_indx = random.sample(range(0, len(self.memory['state'])), batch_size)
            state = self.memory['state'][batch_indx]
            action = self.memory['action'][batch_indx]
            reward = self.memory['reward'][batch_indx]
            next_state = self.memory['next_state'][batch_indx]
            done = self.memory['done'][batch_indx]

            next_values = K.max(self.target_model(next_state), axis=1)
            next_values = next_values * (1 - done)[:, 0]
            target = reward[:, 0] + self.gamma * tf.stop_gradient(next_values)

            self.next_max = K.max(next_values)

            cur_value = self.model(state)
            loss = self._huber_loss(target, K.sum(cur_value * action, axis=1))

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads_and_vars = zip(grads, self.model.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def train(self, env, batch_size=50, total_episodes=200000, output_path=None):

        env.reset()
        state, _, _, _ = env.step_adv(0)
        total_r = 0
        total_psudo_r = 0
        for e in range(total_episodes):
            if e % 2000 == 0:
                save_path = output_path + 'model-' + str(e) + '.h5'
                DLogger.logger().debug('Trained model saved to: ' + save_path)
                self.model.save(save_path)

            action = self.act(state, self.epsilon)
            next_state, reward, done, info = env.step_adv(action)
            self.remember(state, self.actions[action], reward, next_state, done)
            state = next_state
            total_r += reward
            if 'seudo_rew' in info:
                total_psudo_r += info['seudo_rew']
            else:
                total_psudo_r = np.array([0])
            if done[0]:
                env.reset()
                state, _, _, _ = env.step_adv(0)
                DLogger.logger().debug("episode: {}/{}, score: {}, e: {:.2}, reward: {}, seudo-rew:{} max: {} init-act: {}"
                      .format(e, total_episodes, 0, self.epsilon, total_r.mean(),  total_psudo_r.mean(),
                              self.next_max, self.model(state).numpy().mean(axis=0)))
                total_r = 0
                total_psudo_r = 0

            if len(self.memory['state']) > batch_size and e % 100 == 0:
                self.update_target_model()

            if len(self.memory['state']) > batch_size:
                self.replay(batch_size)
