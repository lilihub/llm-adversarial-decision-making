import sys

from data_reader import DataReader
from rnn_learner import RNNAgent
from util.helper import fix_seeds
from util.logger import LogFile, DLogger
import numpy as np
import pandas as pd

local = False

n_folds = 5
confs = []
cells = [2,3,4,5]

for f in range(n_folds):
    for c in range(len(cells)):
        confs.append({'cells': cells[c], 'fold': f})
n_cells = 3
if __name__ == '__main__':

    for fold in range(n_folds):

        output_path = 'trained_model/mrtt_RND_learner/cells_' + str(n_cells) + '/fold_' + str(fold) + '/'

        np.random.seed(1010)
        with LogFile(output_path, 'run.log'):

            # GPT-3.5
            data = DataReader.read_MRTT_RND(data_path = "sim_data/gpt/gpt-3.5-turbo.csv")
            # GPT-4
            # data = DataReader.read_MRTT_RND(data_path = "sim_data/gpt/gpt-4-turbo.csv")

            sh = data['action'].shape

            DLogger.logger().debug('Data dims: ' + str(sh))


            # for synthetic data using their Q learning model
            # data = DataReader.read_nc_data()
            model_path = 'trained_model/inits/cells_' + str(n_cells) + '/model-init.h5'
            agent = RNNAgent(5, 0, n_cells, model_path=model_path)

            indices = np.random.permutation(sh[0])
            folds = np.array_split(indices, n_folds, axis=0)
            # training_idx = np.concatenate(folds[(fold+1):] + folds[:fold])
            # test_idx = folds[fold]
            training_idx, test_idx = np.concatenate([f for j, f in enumerate(folds) if j != fold]), folds[fold]
            ids =   pd.concat([pd.DataFrame({'type': 'train', 'index': training_idx}),
                    pd.DataFrame({'type': 'test', 'index': test_idx})])
            ids.to_csv(output_path + 'indices.csv')
            agent.train(
                data['reward'][training_idx],
                data['action'][training_idx],
                data['state'][training_idx] if 'state' in data else None,
                data['reward'][test_idx],
                data['action'][test_idx],
                data['state'][test_idx] if 'state' in data else None,
                output_path,
                lr=0.001,
            )
