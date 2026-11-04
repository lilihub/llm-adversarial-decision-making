import h5py
import pandas as pd
import numpy as np

from data_process import DataProcess
from util import DLogger


class DataReader:
    def __init__(self):
        pass
    
    @staticmethod
    def read_synth_nc(data_path):
        data = pd.read_csv(data_path, header=0, sep=',', quotechar='"', keep_default_na=False)
        DLogger.logger().debug("read data from " + data_path)
        data['block'] = 1
        ids = data['id'].unique().tolist()
        dftr = pd.DataFrame({'id': ids, 'train': 'train'})
        tdftr = pd.DataFrame({'id': [], 'train': 'test'})
        data = DataProcess.train_test_between_subject(data, pd.concat((dftr, tdftr)), [1], values=['reward', 'action'])
        data = DataProcess.merge_data(data, vals=['reward', 'action'])['merged'][0]
        return data

    @staticmethod
    def read_nc_data(type=None):
        np.random.seed(1010)
        # path = "../data/nc/merged_dynamic.csv"
        path = "/human_data/merged_dynamic.csv"
        data_dynamic = pd.read_csv(path, header=0, sep=',', quotechar='"', keep_default_na=False)

        if type == 'dynamic':
            data = data_dynamic

        # path = "../data/nc/merged_static.csv"
        path = "/human_data/merged_static.csv"
        data_static = pd.read_csv(path, header=0, sep=',', quotechar='"', keep_default_na=False)

        if type == 'static':
            data = data_static

        if type is None:
            data = pd.concat((data_dynamic, data_static))

        ids = data['id'].unique().tolist()
        dftr = pd.DataFrame({'id': ids, 'train': 'train'})
        tdftr = pd.DataFrame({'id': [], 'train': 'test'})
        data = DataProcess.train_test_between_subject(data, pd.concat((dftr, tdftr)), [1], values=['reward', 'action'])
        data = DataProcess.merge_data(data, vals=['reward', 'action'])['merged'][0]

        to_switch_action = np.random.binomial(1, 0.5, data['action'].shape[0])
        data['action'][to_switch_action == 0] = 1 - data['action'][to_switch_action == 0]
        return data


    @staticmethod
    def read_MRTT_Read():
        np.random.seed(1010)
        path = "../data/MRTT/data_Read_summ.csv"
        data = pd.read_csv(path)
        data['block'] = 1
        ids = data['id'].unique().tolist()
        dftr = pd.DataFrame({'id': ids, 'train': 'train'})
        tdftr = pd.DataFrame({'id': [], 'train': 'test'})
        data = DataProcess.train_test_between_subject(data, pd.concat((dftr, tdftr)), [1], values=['reward', 'action'])
        data = DataProcess.merge_data(data, vals=['reward', 'action'])['merged'][0]

        # if you need discrete actions like me :)
        discr_actions = np.floor(np.clip(data['action'] - 0.001, a_min=0, a_max=np.inf) /  4).astype(int)
        data['action'] = discr_actions
        return data


    @staticmethod
    def read_MRTT_RND():
        np.random.seed(1010)
        path = "../data/MRTT/r25_rnd.csv"
        data = pd.read_csv(path)
        data['block'] = 1
        ids = data['id'].unique().tolist()
        dftr = pd.DataFrame({'id': ids, 'train': 'train'})
        tdftr = pd.DataFrame({'id': [], 'train': 'test'})
        data = DataProcess.train_test_between_subject(data, pd.concat((dftr, tdftr)), [1], values=['reward', 'action'])
        data = DataProcess.merge_data(data, vals=['reward', 'action'])['merged'][0]

        # if you need discrete actions like me :)
        discr_actions = np.floor(np.clip(data['action'] - 0.001, a_min=0, a_max=np.inf) /  4).astype(int)
        data['action'] = discr_actions
        return data

if __name__ == '__main__':
    # DataReader.read_nc_data()
    DataReader.read_MRTT_Read()
    pass
