import h5py
import pandas as pd
import numpy as np

from data_process import DataProcess
from util import DLogger


class DataReader:
    def __init__(self, data_path):
        pass

    @staticmethod
    def read_MRTT_RND(data_path):
        np.random.seed(1010)
        path = data_path
        data = pd.read_csv(path)
        data['block'] = 1
        ids = data['id'].unique().tolist()
        dftr = pd.DataFrame({'id': ids, 'train': 'train'})
        tdftr = pd.DataFrame({'id': [], 'train': 'test'})
        data = DataProcess.train_test_between_subject(data, pd.concat((dftr, tdftr)), [1], values=['reward', 'action'])
        data = DataProcess.merge_data(data, vals=['reward', 'action'])['merged'][0]

        # if you need discrete actions like me :)
        discr_actions = np.floor(np.clip(data['action'] - 0.001, a_min=0, a_max=np.inf) /  4).astype(int)
        print(discr_actions)
        data['action'] = discr_actions
        return data

if __name__ == '__main__':
    # DataReader.merge_nc_data()
    # d = DataReader.read_gonogo()
    # DataReader.read_nc_data()
    DataReader.read_MRTT_Read()
    pass
