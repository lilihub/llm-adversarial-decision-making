import tensorflow as tf
from util.helper import multinomial_rvs
from util.logger import LogFile, DLogger
import numpy as np
import pandas as pd

"""
This class takes an joint adversary/learning and simulates them in a gonogo task.
"""
def sim_rmrtt(learner_env, adv_model, output_path, stochastic=False):
    with LogFile(output_path, 'run.log'):
        adv_reward_list = []
        for j in range(0,200):
            total_adv_reward = 0
            learner_env.reset()
            events = []
            adv_state, adv_reward, done, learner_info = learner_env.step_adv(0)
            #
            # cc = []
            # for a in range(learner_env.n_batches):
            #     cc.append(np.random.choice(np.arange(0, 350), 35, replace=False))
            #
            # cc = np.vstack(cc)
            #
            for t in range(10):

                # adv_action = np.argmax(adv_model(adv_state), axis=1)
                # if (np.random.rand() < 0.05):
                #     adv_action = np.random.randint(0, 2, adv_action.shape[0])
                # r_size = int(adv_action.shape[0] * 0.05)
                # adv_action[np.random.choice(np.arange(0, adv_action.shape[0]), r_size)] = np.random.randint(0, 2, r_size)

                if stochastic:
                    adv_action = np.argmax(multinomial_rvs(1, tf.nn.softmax(adv_model(adv_state)).numpy()), axis=1)
                else:
                    # adv_action = adv_model.predict(adv_state)
                    adv_action = np.argmax(adv_model.predict(adv_state), axis=1)

                #
                # adv_action = np.sum(t == cc, axis=1)
                #
                cur_learner_action = learner_info['learner_action'][np.newaxis]
                cur_learner_action_cont = learner_info['learner_action_cont'][np.newaxis]

                # ************************
                cur_learner_action_cont[cur_learner_action_cont == 0] = 0.001
                # adv_action = np.ceil((4 * cur_learner_action_cont - 20) / (6 * cur_learner_action_cont) * 10000)[0]
                adv_action[adv_action <= 0] = 0
                # ************************

                adv_state, adv_reward, done, learner_info = learner_env.step_adv(adv_action)

                events.append({'learner reward': learner_info['learner_reward'],
                               'learner action': cur_learner_action,
                               'learner action cont': cur_learner_action_cont,
                               'adv action': adv_action,
                               'adv reward': adv_reward,
                               'psudo reward': learner_info['seudo_rew'],
                               'adv_state': adv_state})
                t += 1
                total_adv_reward += adv_reward
            adv_reward_list.append(total_adv_reward)   
            pd.DataFrame(events).to_csv(output_path + "events_" + str(j) + ".csv")
        pd.DataFrame(adv_reward_list).to_csv(output_path + "adv_reward_" + ".csv")
