import sys

from data_reader import DataReader
from rnn_learner import RNNAgent
from util.helper import fix_seeds
from util.logger import LogFile, DLogger
import numpy as np
import pandas as pd

if __name__ == '__main__':

    # fix_seeds()
    cells = 3
    output_path = 'trained_model/mrtt_RND_learner_single/cells_' + str(cells) + '/'
    # output_path = 'trained_model/mrtt_RND4_learner_single/cells_' + str(cells) + '/'
    # output_path = 'trained_model/mrtt_human_RND_learner_single/cells_' + str(cells) + '/'
    # output_path = 'trained_model/mrtt_RND_gemini_learner_single/cells_' + str(cells) + '/'
    np.random.seed(1010)
    with LogFile(output_path, 'run.log'):

        # GPT-3.5
        data = DataReader.read_MRTT_RND(data_path = "sim_data/gpt/gpt-3.5-turbo.csv")
        # GPT-4
        # data = DataReader.read_MRTT_RND(data_path = "sim_data/gpt/gpt-4-turbo.csv")

        # Gemini
        # data = DataReader.read_MRTT_RND(data_path = "sim_data/gemini/gemini_1.5.csv")

        sh = data['action'].shape

        DLogger.logger().debug('Data dims: ' + str(sh))


        # for synthetic data using their Q learning model
        # data = DataReader.read_nc_data()
        model_path = 'trained_model/inits/cells_' + str(cells) + '/model-init.h5'
        agent = RNNAgent(5, 0, cells, model_path=model_path)

        agent.train(
            data['reward'],
            data['action'],
            data['state'] if 'state' in data else None,
            data['reward'],
            data['action'],
            data['state'] if 'state' in data else None,
            output_path,
            lr=0.001,
        )
