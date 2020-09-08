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
        self.dataset = dataset
        train_size = int(dataset.size * train_ratio)

        self.data_indices = np.arange(0, dataset.size)
        indices = np.random.choice(self.data_indices, train_size, replace=False)
        self.indices = {
            'train': indices,
            'val': np.array(list(set(self.data_indices) - set(indices)))
        }

    def __len__(self):
        return len(self.data_indices)

    def __str__(self):
        return str(self.dataset)

    def get(self, batch_size, pin_memory=False):
        print(f"{self.__str__} train (size={len(self.indices['train'])})")
        print(f"{self.__str__} val (size={len(self.indices['val'])})")
        return {
            'train': DataLoader(copy.copy(self.dataset(self.indices['train'])),
                                batch_size=batch_size, shuffle=True, pin_memory=pin_memory),
            'val': DataLoader(copy.copy(self.dataset(self.indices['val'])),
                              batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
        }

    def get_all(self, batch_size, pin_memory=False):
        return DataLoader(copy.copy(self.dataset(self.data_indices)),
                          batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
