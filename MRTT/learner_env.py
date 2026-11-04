from util.helper import multinomial_rvs, one_hot
from util.logger import LogFile, DLogger
import numpy as np
import tensorflow as tf
# from tensorflow.python.keras.saving import load_model
from tensorflow.python.keras.models import load_model

"""
This class takes a learner and makes and environment out of it. That is, the learner receives actions of the adversary
uses that action to get reward for the next step, makes an step, and then returns its RNN state and other information
such as number of rewards etc as the state to the adversary.
"""


class LearnverEnv:

    def __init__(self, model, n_actions, n_batches, adv_action_num, mode='earn_max'):
        self.mode = mode
        self.n_batches = n_batches
        self.model = model
        n_cells = model.get_layer('GRU').units
        self.n_actions = n_actions
        self.observation_space = type("observation_space", (object,), {"shape": (2 * self.n_actions + n_cells + 1,)})
        self.action_space = type("action_space", (object,), {"n": self.n_actions})
        self.init_state = np.zeros((self.n_batches, n_cells), dtype=np.float32)
        self.init_reward = np.zeros((self.n_batches,))
        self.adv_action_num = adv_action_num
        self.reset()

    @staticmethod
    def adv_reward(action, reward):
        return np.argmax(action, axis=1)

    @staticmethod
    def adv_done(curr_trial):
        return curr_trial > 10

    def reset(self):
        self.norm_factor = 10
        self.curr_trial = np.zeros((self.n_batches))
        self.learner_rnn_state = self.init_state
        self.total_action = np.zeros((self.n_batches, self.n_actions,), dtype=np.float32)
        self.total_reward = np.zeros((self.n_batches, self.n_actions,), dtype=np.float32)
        self.prev_action = np.zeros((self.n_batches, 5), np.float32)
        self.pred_pol = None
        self.total_inv_earn = np.zeros((self.n_batches,))
        self.total_trustee_earn = np.zeros((self.n_batches,))

    def adv_action_to_reward(self, adv_action):
        if self.pred_pol is None:
            return self.init_reward, self.init_reward
        else:
            return np.floor(3 * (adv_action / (self.adv_action_num - 1)) * self.prev_action_cont), \
                   np.floor(3 * (1 - adv_action / (self.adv_action_num - 1)) * self.prev_action_cont)

    def adv_action_to_reward_fair(self, adv_action):
        if self.pred_pol is None:
            return self.init_reward, self.init_reward
        else:
            repay = (3 * (adv_action / (self.adv_action_num - 1)) * self.prev_action_cont)
            invs_earn = repay + (20 - self.prev_action_cont)
            trustee_earn = (3 * (1 - adv_action / (self.adv_action_num - 1)) * self.prev_action_cont)
            self.total_inv_earn += invs_earn
            self.total_trustee_earn += trustee_earn
            if self.curr_trial[0] < 10:
                return repay, np.zeros((self.n_batches))
            else:
                return repay, -np.abs(self.total_trustee_earn - self.total_inv_earn)

    def discr_to_cont_action(self):
        learner_action_cont = np.zeros((self.n_batches))
        # 0 1 2 3 4   => action 0
        # 5 6 7 8     => action 1
        # 9 10 11 12  => action 2
        # 13 14 15 16 => action 3
        # 17 18 19 20 => action 4
        # for the first action we sample from 0 to 4 (5 different investments)
        learner_action_cont[self.prev_action[:, 0] == 1] = \
            np.random.randint(0, 5, [(self.prev_action[:, 0] == 1).sum()])
        # for the first action we sample from 0 to 4 (5 different investments)
        learner_action_cont[self.prev_action[:, 0] != 1] = \
            5 + (np.argmax(self.prev_action[self.prev_action[:, 0] != 1], 1) - 1) * 4 \
            + np.random.randint(0, 4, [(self.prev_action[:, 0] != 1).sum()])
        return learner_action_cont

    def step_adv(self, adv_action, learner_action=None): #Input: adv_action (initial action is 0), learner_action is None
        if self.mode == 'earn_max':
            learner_reward, adv_reward_kept = self.adv_action_to_reward(adv_action)
        elif self.mode == 'fair_max':
            learner_reward, adv_reward_kept = self.adv_action_to_reward_fair(adv_action)
        else:
            raise Exception('unknown mode')
        self.prev_action = self.step(self.prev_action, learner_reward)
        self.prev_action_cont = self.discr_to_cont_action() # convert discret action to continuous
        if learner_action is not None:
            self.prev_action_cont = learner_action
            # action to bin
            self.prev_action = np.floor(np.clip(learner_action - 0.001, a_min=0, a_max=np.inf) / 4).astype(int)
            self.prev_action = one_hot(self.prev_action, 5)
        return self.get_adv_state(self.prev_action), \
               adv_reward_kept, \
               self.adv_done(self.curr_trial), \
               {'state': self.curr_trial,
                'learner_action': self.prev_action,
                'learner_reward': learner_reward,
                'seudo_rew': np.zeros_like(self.prev_action),
                'learner_action_cont': self.prev_action_cont
                }

    def get_action(self):
        return multinomial_rvs(1, self.pred_pol.numpy())

    def get_adv_state(self, action):
        return np.concatenate([
            self.learner_rnn_state,
            self.pred_pol,
            action,
            # self.total_action / self.norm_factor,
            # self.total_reward / self.norm_factor,
            self.curr_trial[:, np.newaxis] / self.norm_factor,
        ], axis=1)

    def step(self, action, reward): # action and reward here refers to RNN's previous action and reward

        self.curr_trial += 1
        learner_input = np.concatenate([reward[:, np.newaxis, np.newaxis],
                                        action[:, np.newaxis]], axis=2)

        self.learner_rnn_state, self.pred_pol = self.model([learner_input, self.learner_rnn_state])
        self.pred_pol = self.pred_pol[:, -1]
        # print(self.pred_pol.numpy())
        self.learner_rnn_state = self.learner_rnn_state[:, -1, :]
        return self.get_action()
