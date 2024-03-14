import h5py
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class SurvivalDataset(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, datapath, name, transform=None):

        data = np.load(datapath + name + '_Sur.npy')
        # data = np.load(datapath + name + '_death.npy')

        self.image_names2 = data[:,0]
        self.image_names3 = data[:,1]
        self.s = data[:,2].astype(np.int).reshape(-1, 1)
        self.t = data[:,3].astype(np.float).reshape(-1, 1)
        self.transform = transform

        # # normalizes data
        # self._normalize()

        print('=> load {} samples'.format(self.s.shape[0]))

    def __getitem__(self, item):
        X_item2 = np.load(self.image_names2[item])
        X_item3 = np.load(self.image_names3[item])
        e_item = self.s[item]
        y_item = self.t[item]
        pid = self.image_names2[item].split('/')[-1].split('.')[0]
        # print(pid)
        if self.transform:
            X_tensor2 = self.transform(X_item2)
            X_tensor3 = self.transform(X_item3)
        e_tensor = torch.from_numpy(e_item)
        y_tensor = torch.from_numpy(y_item)
        return X_tensor2, X_tensor3 , y_tensor, e_tensor, pid

    def __len__(self):
        return self.s.shape[0]

    def get_y(self):
        aux = [(self.s[i][0],self.t[i][0]) for i in range(self.s.shape[0])]
        ny = np.array(aux, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
        return ny
