from util.logger import LogFile, DLogger
import numpy as np
import pandas as pd

"""
This class takes an joint adversary/learning and simulates them in a bandit task.
"""
def sim_bandit_env(adv_learner, output_path):
    with LogFile(output_path, 'run.log'):
        adv_reward_list = []
        for j in range(0,50):
            DLogger.logger().debug("sim " + str(j))
            adv_learner.reset() # reset adv learner; the initial reward_vec of adv is 00
            events = []

            t = 0
            total_adv_reward = 0
            reward_vec, adv_action, adv_reward, rnn_action, real_model_action = adv_learner.step(np.array([[0, 0]], np.float32)) # input: RNN's or subject's action, output: 
            while t < 100:
                events.append({'r1': reward_vec.numpy()[0, 0],
                               'r2': reward_vec.numpy()[0, 1],
                               'rnn action': rnn_action,
                               'adv action': adv_action.numpy(),
                               'adv reward': adv_reward,
                               'real model action': real_model_action,

                               })
                taken_action = real_model_action if real_model_action is not None else rnn_action
                reward_vec, adv_action, adv_reward, rnn_action, real_model_action = adv_learner.step(
                    np.array(taken_action, np.float32))
                t += 1
                total_adv_reward += adv_reward

            adv_reward_list.append(total_adv_reward)

            pd.DataFrame(events).to_csv(output_path + "events_" + str(j) + ".csv")

        pd.DataFrame(adv_reward_list).to_csv(output_path + "adv_reward_" + ".csv")
