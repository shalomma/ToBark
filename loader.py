import copy
import numpy as np
from torch import manual_seed
from torch.utils.data import DataLoader
from dataset import Dataset


class Loader:
    seed = 14

    def __init__(self, batch_size, pin_memory=False, train_ratio=0.9):
        assert 0 < train_ratio < 1, '0 < r < 1'
        np.random.seed(self.seed)
        manual_seed(self.seed)
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.train_ratio = train_ratio
        self.indices = dict()

    def get(self, dataset: Dataset):
        train_size = int(dataset.size * self.train_ratio)
        data_indices = np.arange(0, dataset.size)
        indices = np.random.choice(data_indices, train_size, replace=False)
        self.indices = {
            'train': indices,
            'val': np.array(list(set(data_indices) - set(indices)))
        }

        print(f"{dataset} train (size={len(self.indices['train'])})")
        print(f"{dataset} val (size={len(self.indices['val'])})")
        return {
            'train': DataLoader(copy.copy(dataset(self.indices['train'])),
                                batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory),
            'val': DataLoader(copy.copy(dataset(self.indices['val'])),
                              batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory)
        }

    def get_all(self, dataset):
        return DataLoader(dataset(np.arange(0, dataset.size)),
                          batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory)
