import multiprocessing
import os
import sys
# sys.path.append('/tf/')
from ddqn import DQNAgent

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from multiprocessing.pool import Pool
import numpy as np
# from tensorflow.python.keras.saving import load_model
from tensorflow.python.keras.models import load_model
from learner_env import LearnverEnv
from util.logger import LogFile, DLogger

configs = []

for b in [200000, 400000]:
    for lr in [0.001, 0.0001, 1e-5]:
        for eps in [0.01, 0.1, 0.2]:
            configs.append({'b': b, 'lr': lr, 'eps': eps})


def run_adv():
    # fix_seeds()

    np.set_printoptions(precision=3)
# index 9 400,000 0.01, 0.001
    buf = configs[9]['b']
    lr = configs[9]['lr']
    eps = configs[9]['eps']
    
    # # GPT-3.5
    # learner_model_path = 'trained_model/RNN_learner_single/cells_5gpt-3.5-turbo/model-8600.h5'
    # output_path =  'trained_model/adv_RL_' + str(buf)+ '_eps_' + str(eps) + '_lr_' + str(lr) + '/' + 'gpt-3.5-turbo/'

    # GPT-4
    # learner_model_path = 'trained_model/RNN_learner_single/cells_5gpt-4-turbo/model-1000.h5'
    # output_path =  'trained_model/adv_RL_' + str(buf)+ '_eps_' + str(eps) + '_lr_' + str(lr) + '/' + 'gpt-4-turbo/'

    # Gemini-1.5
    # learner_model_path = 'trained_model/RNN_learner_single/cells_5gemini_1.5/model-2000.h5'
    # output_path =  'trained_model/adv_RL_' + str(buf)+ '_eps_' + str(eps) + '_lr_' + str(lr) + '/' + 'gemini_1.5/'

    # DeepSeek-chat
    learner_model_path = 'trained_model/RNN_learner_single/cells_5deepseek-chat/model-1300.h5'
    output_path =  'trained_model/adv_RL_' + str(buf)+ '_eps_' + str(eps) + '_lr_' + str(lr) + '/' + 'deepseek-chat/'

    DLogger.logger().debug('config: ')
    np.set_printoptions(precision=3)

    with LogFile(output_path, 'run.log'):
        DLogger.logger().debug("Learner model loaded from path {}".format(learner_model_path))
        model = load_model(learner_model_path)
        DLogger.logger().debug("Learner model:")
        model.summary(print_fn=DLogger.logger().debug)
        le = LearnverEnv(model, 2, 1000)
        le.reset()
        agent = DQNAgent(le.observation_space.shape[0], 4, buf, epsilon=eps, lr=lr)
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

    # run(run_adv, ncpus, chunk)
    run_adv()
