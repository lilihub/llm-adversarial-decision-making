import os
from multiprocessing.pool import Pool
from adv_learner_gpt import AdvLearner_gpt
from sim_bandit import sim_bandit_env
from util import DLogger



"""
In this file trained learner in QRL and trained adversary for that QRL are loaded 
and then simulated in the task each 400 times to assess the power of adversary against the trained RNN.

Note that the learner is a trained RNN to provide state and other information to the adversary, 
but the actual environment used for simulations is a Q learning using the same parameters 
used in NC paper.   
"""
# GPT-3.5
# learner_path = 'trained_model/RNN_learner_single/cells_5gpt-3.5-turbo/model-8600.h5'
# adv_path = 'trained_model/adv_RL_400000_eps_0.01_lr_0.001/gpt-3.5-turbo/model-100000.h5'

# GPT-4
# learner_path = 'trained_model/RNN_learner_single/cells_5gpt-4-turbo/model-1000.h5'
# adv_path = 'trained_model/adv_RL_400000_eps_0.01_lr_0.001/gpt-4-turbo/model-100000.h5'

# Gemini_1.5
learner_path = 'trained_model/RNN_learner_single/cells_5gemini_1.5/model-5000.h5'
adv_path = 'trained_model/adv_RL_400000_eps_0.01_lr_0.001/gemini_1.5/model-100000.h5'

def run_sim():
    # try:
    #     adv_path = dirs_iter[i][1] + '/model-' + dirs_iter[i][2] + '.h5'
    #     output_path = '/evaluate_model/RNN_nc_sim' + dirs_iter[i][2] + '/' + dirs_iter[i][0] + '/'
    #     bandit_env = AdvLearner(learner_path, adv_path, output_path, None)
    #     sim_bandit_env(bandit_env, output_path)
    # except:
    #     DLogger.logger().debug("Exception for " + str(i))

    # GPT-3.5
    # output_path = 'evaluate_model/ADV_vs_RNN/gpt-3.5-turbo/'
    # GPT-4
    # output_path = 'evaluate_model/ADV_vs_RNN/gpt-4-turbo/'
    # Gemini_1.5
    output_path = 'evaluate_model/ADV_vs_RNN/gemini_1.5/'

    bandit_env = AdvLearner_gpt(learner_path, adv_path, output_path, None)
    sim_bandit_env(bandit_env, output_path)


# def run(f, n_proc):
#     p = Pool(n_proc)
#     p.map(f, range(len(dirs_iter)))
#     p.close()  # no more tasks
#     p.join()  # wrap up current tasks


if __name__ == '__main__':
    # run(run_sim, 20)
     run_sim()
