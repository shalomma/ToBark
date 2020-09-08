import copy
import numpy as np
from torch import manual_seed
from torch.utils.data import DataLoader
from dataset import Dataset


class Loader:
    seed = 14

    def __init__(self, dataset: Dataset, train_ratio=0.9):
        assert 0 < train_ratio < 1, '0 < r < 1'
        np.random.seed(self.seed)
        manual_seed(self.seed)
        train_size = int(dataset.size * train_ratio)
        data_indices = np.arange(0, dataset.size)
        indices = np.random.choice(data_indices, train_size, replace=False)
        self.indices = {
            'train': indices,
            'val': np.array(list(set(data_indices) - set(indices)))
        }
        self.data = {
            'train': copy.copy(dataset(self.indices['train'])),
            'val': copy.copy(dataset(self.indices['val']))
        }

    def __len__(self):
        return len(self.data['train']) + len(self.data['val'])

    def __str__(self):
        return str(self.data['train'])

    def get(self, batch_size, pin_memory=False):
        loaders = dict()
        loaders['train'] = DataLoader(self.data['train'], batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
        loaders['val'] = DataLoader(self.data['val'], batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
        print(f"{self.data['train']} train (size={len(self.data['train'])})")
        print(f"{self.data['val']} val (size={len(self.data['val'])})")
        return loaders

    def get_all(self, batch_size, pin_memory=False):
        return DataLoader(self.data['train'] + self.data['val'],
                          batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
