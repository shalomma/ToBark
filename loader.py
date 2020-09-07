from torch import manual_seed
from torch.utils.data import DataLoader
import dataset
import numpy as np


class Loader:
    seed = 14

    def __init__(self, size, train_ratio):
        np.random.seed(self.seed)
        manual_seed(self.seed)
        self.data = dict()
        train_size = int(size * train_ratio)
        data_indices = np.arange(0, size)
        indices = np.random.choice(data_indices, train_size, replace=False)
        self.indices = {
            'train': indices,
            'val': np.array(list(set(data_indices) - set(indices)))
        }

    def __len__(self):
        return len(self.data['train']) + len(self.data['val'])

    def __str__(self):
        return str(self.data['train'])

    def get(self, batch_size, pin_memory=False):
        loaders = dict()
        loaders['train'] = DataLoader(self.data['train'], batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
        print(f"{self.data['train']} train (size={len(self.data['train'])})")
        if self.indices['val'].size > 0:
            loaders['val'] = DataLoader(self.data['val'], batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
            print(f"{self.data['val']} val (size={len(self.data['val'])})")
        return loaders

    def get_all(self, batch_size, pin_memory=False):
        return DataLoader(self.data['train'] + self.data['val'],
                          batch_size=batch_size, shuffle=True, pin_memory=pin_memory)


class UrbanSound8KLoader(Loader):
    def __init__(self, train_ratio=0.9):
        size = dataset.UrbanSound8K.size
        super(UrbanSound8KLoader, self).__init__(size, train_ratio)
        self.data['train'] = dataset.UrbanSound8K(self.indices['train'])
        self.data['val'] = dataset.UrbanSound8K(self.indices['val'])


class CatAndDogsLoader(Loader):
    def __init__(self, train_ratio=0.9):
        size = dataset.CatsAndDogs.size
        super(CatAndDogsLoader, self).__init__(size, train_ratio)
        self.data['train'] = dataset.CatsAndDogs(self.indices['train'])
        self.data['val'] = dataset.CatsAndDogs(self.indices['val'])


class MelSpecEncodedLoader(Loader):
    def __init__(self, prefix, size, train_ratio=0.9):
        super(MelSpecEncodedLoader, self).__init__(size, train_ratio)
        self.data['train'] = dataset.MelSpecEncoded(prefix, self.indices['train'])
        self.data['val'] = dataset.MelSpecEncoded(prefix, self.indices['val'])
