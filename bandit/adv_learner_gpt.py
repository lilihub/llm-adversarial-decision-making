import numpy as np
from tensorflow.python.keras.backend import argmax
from tensorflow.python.keras.models import load_model

from learner_env import LearnverEnv
from gpt_env import GPT_ENV
from sim_bandit import sim_bandit_env
from util.logger import LogFile, DLogger
import tensorflow as tf
import tensorflow.keras.backend as K

"""
This class includes an adversary and an a learning in the form of a RNN, and also 
a real agent which selects the actions. In each step, the model receives the action 
of the learner, return the reward vector provided by the adversary and also action of 
the RNN agent and also action of the real agent.
"""
class AdvLearner_gpt:

    def __init__(self, learner_model_path, adv_model_path, output_path, real_model):
        np.set_printoptions(precision=5)
        self.real_model = real_model # subject
        self.events = []
        with LogFile(output_path, 'run.log'):
            if real_model is None:
                DLogger.logger().debug("Real model is not provided -- using learner for selecting actions.")
            DLogger.logger().debug("Learner model loaded from path {}".format(learner_model_path))
            learner_model = load_model(learner_model_path, compile=False)
            self.le = LearnverEnv(learner_model, 2, 1)
            DLogger.logger().debug("Adv model loaded from path {}".format(adv_model_path))
            self.adv_model = load_model(adv_model_path, compile=False)
        self.reset()

    def step(self, action): # action here refers to RNN's action or subject's action
        reward = self.le.step_vec(action, self.reward_vec) # reward_vec here refers to adv's reward_vec; the initial reward_vec is 00; the initial RNN action is 00; step RNN (updating RNN states, predicted policies, total reward and action) and reformat adv's reward_vec
        rnn_action = self.le.get_action() # RNN makes an action
        logits = self.adv_model.predict(self.le.get_adv_state()) # after stepping RNN, get inputs for adv: RNN's state, predicted policy, total action (RNN's or subject's), total reward (given by adv), and current trial, and then let adv_model to predict

        if len(logits) > 1:  #if the return includes both policies and estiamted values
            logits = logits[0]
        adv_action = argmax(logits, axis=1) # adv makes an action
        self.reward_vec = self.le.adv_action_to_reward(adv_action.numpy())

        if self.real_model is not None:
            # real_action = self.real_model.step(action, reward)[np.newaxis]
            treasure = self.reward_vec.numpy()[np.arange(self.reward_vec.shape[0]), (action[:, 1]).astype(np.int32)]
            real_action = self.real_model.step(action, treasure)[np.newaxis]
        else:
            real_action = None

        return self.reward_vec, adv_action, self.le.adv_reward(action, reward), rnn_action, real_action
    # return adv's reward vec, adv's action, RNN's action, which means adv's reward, RNN's action, real_action

    def reset(self):
        self.le.reset()
        if self.real_model:
            self.real_model.reset()
        self.reward_vec = K.zeros((1, 2))


if __name__ == '__main__':
    output_path = '../nongit/results/temp/'
    bandit_env = AdvLearner(
                '../nongit/archive/learner/nc/learner_human_nc/learner_nc_cells_5/model-8800.h5',
                '../nongit/archive/RL/nc/results_human_nc/RL_nc_5cells_1layers_0.5ent_128units/model-140000.h5',
                output_path,
                GPT_ENV()
                 )
    sim_bandit_env(bandit_env, output_path)
