import numpy as np


class DataProcess:

    def __init__(self):
        pass


    @staticmethod
    def train_test_between_subject(data,
                                   indx_data,
                                   train_blocks,
                                   values=None
                                   ):
        if values is None:
            values = ['reward', 'action', 'state']
        train = {}
        sdata = indx_data.loc[indx_data.train == "train"]

        ids = sdata['id'].unique().tolist()

        for s_id in ids:
            sub_data = data.loc[data.id == s_id]
            train[s_id] = []

            for t in train_blocks:
                sub_data_t = sub_data.loc[sub_data.block.isin([t])]
                dicts = {}
                for v in values:
                    if v in sub_data:
                        dicts[v] = sub_data_t[v].values[np.newaxis]
                dicts['id'] = s_id
                dicts['block'] = t
                train[s_id].append(dicts)
        return train

    @staticmethod
    def get_max_seq_len(train):
        max_len = -np.Inf
        max_fmri = -np.Inf
        for s_data in train.values():
            for t_data in s_data:
                if t_data['reward'].shape[1] > max_len:
                    max_len = t_data['reward'].shape[1]

                if not 'fmri_timeseries'in t_data or t_data['fmri_timeseries'] is None:
                    max_fmri = None
                else:
                    if t_data['fmri_timeseries'].shape[1] > max_fmri:
                        max_fmri = t_data['fmri_timeseries'].shape[1]

        return max_len, max_fmri

    @staticmethod
    def merge_blocks(data):
        merged_data = {}

        for k in sorted(data.keys()):
            v = data[k]
            merged_data[k] = DataProcess.merge_data({'merged': v})['merged']

        return merged_data

    @staticmethod
    def merge_data(train, batch_size=-1, vals=None):

        if vals is None:
            vals = ['action', 'reward', 'state']
        max_len, _ = DataProcess.get_max_seq_len(train)

        def app_not_None(arr, to_append, max_len):
            if to_append is not None:
                if max_len is not None:
                    pad_shape = [(0, 0), (0, (max_len - to_append.shape[1]))] + [(0,0)] * (len(to_append.shape)-2)
                    arr.append(np.lib.pad(to_append, pad_shape, 'constant', constant_values=(0, -1)))
                else:
                    arr.append(to_append)

        def none_if_empty(arr):
            if len(arr) > 0:
                return np.concatenate(arr)
            return None

        dicts = {}
        for v in vals:
            dicts[v] = []
        ids = []
        seq_lengths = []

        batches = []

        cur_size = 0

        for k_data in reversed(sorted(train.keys())):
            s_data = train[k_data]
            for t_data in s_data:
                seq_lengths.append(t_data[list(t_data.keys())[0]].shape[1])
                for v in vals:
                    app_not_None(dicts[v], t_data[v], max_len)
                ids.append(t_data['id'])

                cur_size += 1
                if batch_size != -1 and cur_size >= batch_size:
                    out_dict = {}
                    for v in vals:
                        out_dict[v] = none_if_empty(dicts[v])
                    out_dict['block'] = len(batches)
                    out_dict['id'] = str(ids)
                    batches.append(out_dict)

                    dicts = {}
                    for v in vals:
                        dicts[v] = []
                    ids = []
                    cur_size = 0

        if cur_size > 0:
            out_dict = {}
            for v in vals:
                out_dict[v] = none_if_empty(dicts[v])
            out_dict['block'] = len(batches)
            out_dict['id'] = ids
            out_dict['seq_lengths'] = np.array(seq_lengths)
            batches.append(out_dict)

        return {'merged': batches}
