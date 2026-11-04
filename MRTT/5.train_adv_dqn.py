import multiprocessing
import os

from learner_env import LearnverEnv
from ddqn import DQNAgent

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from multiprocessing.pool import Pool
import numpy as np
# from tensorflow.python.keras.saving import load_model
from tensorflow.python.keras.models import load_model
from util.logger import LogFile, DLogger

configs = []

for b in [200000, 400000]:
    for lr in [0.001, 0.0001, 1e-5]:
        for eps in [0.01, 0.1, 0.2]:
            configs.append({'b': b, 'lr': lr, 'eps': eps})

# 00:{'b': 200000, 'lr': 0.001, 'eps': 0.01}
# 01:{'b': 200000, 'lr': 0.001, 'eps': 0.1}
# 02:{'b': 200000, 'lr': 0.001, 'eps': 0.2}
# 03:{'b': 200000, 'lr': 0.0001, 'eps': 0.01}
# 04:{'b': 200000, 'lr': 0.0001, 'eps': 0.1}
# 05:{'b': 200000, 'lr': 0.0001, 'eps': 0.2}
# 06:{'b': 200000, 'lr': 1e-05, 'eps': 0.01}
# 07:{'b': 200000, 'lr': 1e-05, 'eps': 0.1}
# 08:{'b': 200000, 'lr': 1e-05, 'eps': 0.2}
# 09:{'b': 400000, 'lr': 0.001, 'eps': 0.01}
# 10:{'b': 400000, 'lr': 0.001, 'eps': 0.1}
# 11:{'b': 400000, 'lr': 0.001, 'eps': 0.2}
# 12:{'b': 400000, 'lr': 0.0001, 'eps': 0.01}
# 13:{'b': 400000, 'lr': 0.0001, 'eps': 0.1}
# 14:{'b': 400000, 'lr': 0.0001, 'eps': 0.2}
# 15:{'b': 400000, 'lr': 1e-05, 'eps': 0.01}
# 16:{'b': 400000, 'lr': 1e-05, 'eps': 0.1}
# 17:{'b': 400000, 'lr': 1e-05, 'eps': 0.2}

def run_adv():
    # fix_seeds()

    np.set_printoptions(precision=3)
    i= 9 #17
    buf = configs[i]['b']
    lr = configs[i]['lr']
    eps = configs[i]['eps']
    mode = 'earn_max'

    # DeepSeek-chat MAX i=11; FAIR i=17; iteration = 3000
    learner_model_path = 'trained_model/mrtt_RND_deepseek_learner_single/cells_3/model-6000.h5'
    output_path =  'trained_model/RL_rmrtt_dqn_RND_deepseek_' + str(mode) +  '/RL_nc_dqn_buf_' + str(buf)+ '_eps_' + str(eps) + '_lr_' + str(lr) + '/'

    # GPT-3.5 MAX i=13; FAIR i=13; iteration = 3000 when i=9 for both 
    # learner_model_path = 'trained_model/mrtt_RND_learner_single/cells_3/model-3000.h5'
    # output_path =  'trained_model/RL_rmrtt_dqn_RND_' + str(mode) +  '/RL_nc_dqn_buf_' + str(buf)+ '_eps_' + str(eps) + '_lr_' + str(lr) + '/'
    # GPT-4 MAX i=11; FAIR i=17
    # learner_model_path = 'trained_model/mrtt_RND4_learner_single/cells_3/model-5000.h5'
    # output_path =  'trained_model/RL_rmrtt_dqn_RND4_' + str(mode) +  '/RL_nc_dqn_buf_' + str(buf)+ '_eps_' + str(eps) + '_lr_' + str(lr) + '/'

    # Gemini MAX i=9; FAIR i=9; iteration = 8000 for both
    # learner_model_path = 'trained_model/mrtt_RND_gemini_learner_single/cells_3/model-8000.h5'
    # output_path =  'trained_model/RL_rmrtt_dqn_RND_gemini_' + str(mode) +  '/RL_nc_dqn_buf_' + str(buf)+ '_eps_' + str(eps) + '_lr_' + str(lr) + '/'

    # Human
    # learner_model_path = 'trained_model/mrtt_human_RND_learner_single/cells_3/model-700.h5'
    # output_path =  'trained_model/RL_human_rmrtt_dqn_RND_' + str(mode) +  '/RL_nc_dqn_buf_' + str(buf)+ '_eps_' + str(eps) + '_lr_' + str(lr) + '/'

    DLogger.logger().debug('config: ')
    np.set_printoptions(precision=3)
    adv_action_num = 5

    with LogFile(output_path, 'run.log'):
        DLogger.logger().debug("Learner model loaded from path {}".format(learner_model_path))
        model = load_model(learner_model_path)
        DLogger.logger().debug("Learner model:")
        model.summary(print_fn=DLogger.logger().debug)
        le = LearnverEnv(model, 5, 1000, adv_action_num, mode=mode)
        le.reset()
        agent = DQNAgent(le.observation_space.shape[0], adv_action_num, buf, epsilon=eps, lr=lr)
        agent.train(env=le, output_path=output_path, batch_size=1000, total_episodes=int(1e10))


def run(f, n_proc, chunk):
    p = Pool(n_proc)
    start = min(len(configs), (chunk - 1) * n_proc)
    end = min(len(configs), chunk * n_proc)
    p.map(f, range(start, end))
    p.close()  # no more tasks
    p.join()  # wrap up current tasks


if __name__ == '__main__':

    # run(run_adv, 1)
    #
    # if len(sys.argv) == 2:
    #     chunk = int(sys.argv[1])
    # else:
    #     raise Exception("invalid processing chunk")
    # try:
    #     ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    #     print("SLURM_JOB_CPUS_PER_NODE" + " found")
    #     print("CPU: " +str(ncpus))
    # except KeyError:
    #     ncpus = multiprocessing.cpu_count()
    #     print("SLURM_JOB_CPUS_PER_NODE" + " not found")
    #     print("CPU: " +str(ncpus))

    # run(run_adv, 1, chunk)
    run_adv()