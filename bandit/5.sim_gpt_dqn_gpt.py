import os
from multiprocessing.pool import Pool
from adv_learner_gpt import AdvLearner_gpt
from sim_bandit import sim_bandit_env
from deepseek_env import Deepseek_ENV
from util import DLogger
import sys
# sys.path.append('/tf/')


"""
In this file trained learner in QRL and trained adversary for that QRL are loaded 
and then simulated in the task each 400 times to assess the power of adversary against the trained RNN.

Note that the learner is a trained RNN to provide state and other information to the adversary, 
but the actual environment used for simulations is a Q learning using the same parameters 
used in NC paper.   
"""
# GPT-3.5-turbo
# learner_path = 'trained_model/RNN_learner_single/cells_5gpt-3.5-turbo/model-8600.h5'
# adv_path = 'trained_model/adv_RL_400000_eps_0.01_lr_0.001/gpt-3.5-turbo/model-100000.h5'

# GPT-4-turbo
# learner_path = 'trained_model/RNN_learner_single/cells_5gpt-4-turbo/model-1000.h5'
# adv_path = 'trained_model/adv_RL_400000_eps_0.01_lr_0.001/gpt-4-turbo/model-100000.h5'

# Gemini_1.5
# learner_path = 'trained_model/RNN_learner_single/cells_5gemini_1.5/model-2000.h5'
# adv_path = 'trained_model/adv_RL_400000_eps_0.01_lr_0.001/gemini_1.5/model-100000.h5'

# Deepseek-chat
learner_path = 'trained_model/RNN_learner_single/cells_5deepseek-chat/model-1300.h5'
adv_path = 'trained_model/adv_RL_400000_eps_0.01_lr_0.001/deepseek-chat/model-100000.h5'

def run_sim():
    # try:
    # GPT-3.5
    # output_path = 'evaluate_model/RNN_adv_sim_400000_eps_0.01_lr_0.001/gpt-3.5-turbo/'
    # bandit_env = AdvLearner_gpt(learner_path, adv_path, output_path, GPT_ENV(GPT_version='GPT-3.5-turbo-1106'))

    # GPT-4
    # output_path = 'evaluate_model/RNN_adv_sim_400000_eps_0.01_lr_0.001/gpt-4-turbo/'
    # bandit_env = AdvLearner_gpt(learner_path, adv_path, output_path, GPT_ENV(GPT_version='GPT-4-turbo'))

    # Gemini_1.5
    # output_path = 'evaluate_model/RNN_adv_sim_400000_eps_0.01_lr_0.001/gemini_1.5/'
    # bandit_env = AdvLearner_gpt(learner_path, adv_path, output_path, Gemini_ENV(Gemini_version='gemini-1.5-flash'))

    
    # Deepseek-chat
    output_path = 'evaluate_model/RNN_adv_sim_400000_eps_0.01_lr_0.001/deepseek-chat/'
    bandit_env = AdvLearner_gpt(learner_path, adv_path, output_path, Deepseek_ENV(Deepseek_version='deepseek-chat'))

    sim_bandit_env(bandit_env, output_path)


# def run(f, n_proc):
#     p = Pool(n_proc)
#     p.map(f, range(len(dirs_iter)))
#     p.close()  # no more tasks
#     p.join()  # wrap up current tasks


if __name__ == '__main__':
    # run(run_sim, 40)
    run_sim()
