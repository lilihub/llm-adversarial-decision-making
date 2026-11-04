import sys

from data_reader import DataReader
from rnn_learner import RNNAgent
from util.helper import fix_seeds
from util.logger import LogFile, DLogger
import numpy as np
import pandas as pd


if __name__ == '__main__':

    # fix_seeds()
    n_cells = 5
    # output_path = 'trained_model/RNN_learner_single/cells_' + str(n_cells)  + 'gpt-3.5-turbo/'
    # output_path = 'trained_model/RNN_learner_single/cells_' + str(n_cells)  + 'gpt-4-turbo/'
    # output_path = 'trained_model/RNN_learner_single/cells_' + str(n_cells)  + 'gemini_1.5/'
    output_path = 'trained_model/RNN_learner_single/cells_' + str(n_cells)  + 'deepseek-chat/'
    np.random.seed(1010)
    with LogFile(output_path, 'run.log'):
        # data = DataReader.read_synth_nc(data_path = "sim_data/gpt/gpt-3.5-turbo.csv")
        # data = DataReader.read_synth_nc(data_path = "sim_data/gpt/gpt-4-turbo.csv")
        # data = DataReader.read_synth_nc(data_path = "sim_data/gemini/gemini_1.5.csv")
        data = DataReader.read_synth_nc(data_path = "sim_data/deepseek/deepseek-chat.csv")
        sh = data['action'].shape

        DLogger.logger().debug('Data dims: ' + str(sh))

        # for synthetic data using their Q learning model
        # data = DataReader.read_nc_data()
        model_path = 'trained_model/inits/learner_nc_cells_' + str(n_cells) + '/model-init.h5'
        agent = RNNAgent(2, 0, n_cells, model_path=model_path)

        agent.train(
            data['reward'],
            data['action'],
            data['state'] if 'state' in data else None,
            data['reward'],
            data['action'],
            data['state'] if 'state' in data else None,
            output_path,
            lr=0.005
        )
