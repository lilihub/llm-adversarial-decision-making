import numpy as np

class ArrQ:

    def __init__(self, size, shape):
        self.q = np.zeros(shape=[size] + shape)
        self.cur_index = 0
        self.size = size

    def append(self, d):
        end_index = self.cur_index + d.shape[0]
        if end_index < self.size:
            self.q[self.cur_index:end_index, :] = d
            self.cur_index += d.shape[0]
        else:

            self.q[:-d.shape[0], ] = \
                self.q[(d.shape[0] - (self.size - self.cur_index)):self.cur_index]
            self.q[-d.shape[0]:] = d

            self.cur_index = self.size

    def __len__(self):
        return self.cur_index

    def __getitem__(self, item):
        return self.q[item]
