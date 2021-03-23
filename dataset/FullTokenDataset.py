import json
import os
from torch.utils.data import Dataset
from .IndexedDataset import make_dataset,MMapIndexedDataset
import random
import numpy as np

class FullTokenDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.max_len = config.getint('train', 'max_len')
        self.indexed_dataset = make_dataset(config, mode)
        self.data_num = len(self.indexed_dataset)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            sent = self.indexed_dataset[idx]
            while sent.shape[0] < self.max_len - 30:
                ridx = random.randint(0, self.data_num-1)
                rsent = self.indexed_dataset[ridx]
                sent = np.concatenate([sent, rsent])[:self.max_len]
            return sent
        elif isinstance(idx, slice):
            sents = self.indexed_dataset[idx]
            return sents

    def __len__(self):
        return len(self.indexed_dataset)

class MultiDocDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.max_len = config.getint('train', 'max_len')
        path = config.get('data', '%s_data' % mode)
        flist = config.get('data', '%s_files' % mode).split(',')
        self.datasets = [MMapIndexedDataset(os.path.join(path, f), False) for f in flist]
        self.lens = [len(d) for d in self.datasets]
        self.length = sum(self.lens)
        self.idlist = np.arange(0, self.length)
        np.random.shuffle(self.idlist)
    
    def __getitem__(self, idx):
        ridx = int(self.idlist[idx])
        for i in range(len(self.lens)):
            if ridx > self.lens[i]:
                ridx -= self.lens[i]
            else:
                return self.datasets[i][ridx]
        raise ValueError('Index is larger than the number of data')
    
    def __len__(self):
        return self.length