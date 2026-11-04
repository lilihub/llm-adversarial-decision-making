import os
from multiprocessing.pool import Pool
from tensorflow.python.keras.models import load_model
from learner_env1 import LearnverEnv
# sim_rmrtt for simulating GPT
# sim_rmrtt1 for simulating GPT learner
from sim_rmrtt import sim_rmrtt
# from gpt_env import GPT_ENV
# from gemini_env import Gemini_ENV
from deepseek_env import Deepseek_ENV
# from nc.adv_learner import AdvLearner
# from nc.sim_bandit import sim_bandit_env
from util import DLogger
import numpy as np



"""
In this file trained learner in QRL and trained adversary for for that QRL are loaded 
and then simulated in the task each 400 times to assess the power of adversary against the trained RNN.

Note that the learner is a trained RNN to provide state and other information to the adversary, 
but the actual environment used for simulations is a Q learning using the same parameters 
used in NC paper.   
"""
# GPT-3.5
# learner_path = 'trained_model/mrtt_RND_learner_single/cells_3/model-3000.h5'
# GPT-4
# learner_path = 'trained_model/mrtt_RND4_learner_single/cells_3/model-5000.h5'
# Gemini_1.5
# learner_path = 'trained_model/mrtt_RND_gemini_learner_single/cells_3/model-8000.h5'
# DeepSeek
learner_path = 'trained_model/mrtt_RND_deepseek_learner_single/cells_3/model-6000.h5'
# Human
# learner_path = 'trained_model/mrtt_human_RND_learner_single/cells_3/model-700.h5'


def run_sim():
    # try:
        # GPT-3.5 MAX 
        # mode = 'earn_max'
        # adv_path = 'trained_model/RL_rmrtt_dqn_RND_earn_max/RL_nc_dqn_buf_400000_eps_0.1_lr_0.0001/model-176000.h5'
        # output_path = 'evaluate_model/earn_max/RL_nc_dqn_buf_400000_eps_0.1_lr_0.0001/'
        # adv_path = 'trained_model/RL_rmrtt_dqn_RND_earn_max/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/model-40000.h5'
        # output_path = 'evaluate_model/earn_max/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/'
        # GPT-3.5 FAIR
        # mode = 'earn_max'
        # adv_path = 'trained_model/RL_rmrtt_dqn_RND_fair_max/RL_nc_dqn_buf_400000_eps_0.1_lr_0.0001/model-300000.h5'
        # output_path = 'evaluate_model/fair_max/RL_nc_dqn_buf_400000_eps_0.1_lr_0.0001/'
        # adv_path = 'trained_model/RL_rmrtt_dqn_RND_fair_max/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/model-42000.h5'
        # output_path = 'evaluate_model/fair_max/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/'

        # GPT-4 MAX
        # mode = 'earn_max'
        # adv_path = 'trained_model/RL_rmrtt_dqn_RND4_earn_max/RL_nc_dqn_buf_400000_eps_0.2_lr_0.001/model-20000.h5'
        # output_path = 'evaluate_model/earn_max4/RL_nc_dqn_buf_400000_eps_0.2_lr_0.001/'
        # adv_path = 'trained_model/RL_rmrtt_dqn_RND4_earn_max/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/model-22000.h5'
        # output_path = 'evaluate_model/earn_max4/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/'

        # GPT-4 FAIR
        # mode = 'earn_max' #still use earn_max even though it's simulating fair situation
        # adv_path = 'trained_model/RL_rmrtt_dqn_RND4_fair_max/RL_nc_dqn_buf_400000_eps_0.2_lr_1e-05/model-50000.h5'
        # output_path = 'evaluate_model/fair_max4/RL_nc_dqn_buf_400000_eps_0.2_lr_1e-05/'
        # adv_path = 'trained_model/RL_rmrtt_dqn_RND4_fair_max/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/model-70000.h5'
        # output_path = 'evaluate_model/fair_max4/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/'
       

        # Gemini_1.5 MAX
        # mode = 'earn_max'
        # adv_path = 'trained_model/RL_rmrtt_dqn_RND_gemini_earn_max/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/model-84000.h5'
        # output_path = 'evaluate_model/earn_max_gemini/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/'
        # Gemini_1.5 FAIR
        # mode = 'earn_max'
        # adv_path = 'trained_model/RL_rmrtt_dqn_RND_gemini_fair_max/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/model-34000.h5'
        # output_path = 'evaluate_model/fair_max_gemini/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/'

        # DeepSeek MAX
        mode = 'earn_max'
        adv_path = 'trained_model/RL_rmrtt_dqn_RND_deepseek_earn_max/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/model-36000.h5'
        output_path = 'evaluate_model/earn_max_deepseek/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/'
        # DeepSeek FAIR
        # mode = 'earn_max'
        # adv_path = 'trained_model/RL_rmrtt_dqn_RND_gemini_fair_max/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/model-34000.h5'
        # output_path = 'evaluate_model/fair_max_gemini/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/'


        # Human MAX
        # mode = 'earn_max'
        # adv_path = 'trained_model/RL_human_rmrtt_dqn_RND_earn_max/RL_nc_dqn_buf_400000_eps_0.2_lr_1e-05/model-100000.h5'
        # output_path = 'evaluate_model/human_earn_max/RL_nc_dqn_buf_400000_eps_0.2_lr_1e-05/'
        # Human FAIR
        # mode = 'fair_max'
        # adv_path = 'trained_model/RL_human_rmrtt_dqn_RND_fair_max/RL_nc_dqn_buf_200000_eps_0.1_lr_0.001/model-400000.h5'
        # output_path = 'evaluate_model/human_fair_max/RL_nc_dqn_buf_200000_eps_0.1_lr_0.001/'

        DLogger.logger().debug("Learner model loaded from path {}".format(learner_path))
        learner_model = load_model(learner_path, compile=False)
        # GPT-3.5
        # le = LearnverEnv(learner_model, 5, 1, 5, mode=mode, real_model = GPT_ENV(GPT_version = 'gpt-3.5-turbo-1106'))
        # GPT-4
        # le = LearnverEnv(learner_model, 5, 1, 5, mode=mode, real_model = GPT_ENV(GPT_version = 'gpt-4-turbo'))
        # Gemini_1.5
        # le = LearnverEnv(learner_model, 5, 1, 5, mode=mode, real_model = Gemini_ENV(Gemini_version = 'gemini-1.5-flash'))
        # DeepSeek
        le = LearnverEnv(learner_model, 5, 1, 5, mode=mode, real_model = Deepseek_ENV(Deepseek_version = 'deepseek-chat'))


        DLogger.logger().debug("Adv model loaded from path {}".format(adv_path))
        adv_model = load_model(adv_path, compile=False)
        sim_rmrtt(le, adv_model, output_path)


def run(f, n_proc):
    p = Pool(n_proc)
    p.map(f, range(len(dirs_iter)))
    p.close()  # no more tasks
    p.join()  # wrap up current tasks


if __name__ == '__main__':
    # run(run_sim, 40)
    # for i in range(len(dirs_iter)):
    run_sim()

