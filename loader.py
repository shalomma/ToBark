from torch import manual_seed
from torch.utils.data import DataLoader
import dataset
import numpy as np


class Loader:
    seed = 14

    def __init__(self):
        manual_seed(self.seed)
        self.data = dict()
        size = 8732
        val_size = 800
        data_indices = np.arange(0, size)
        indices = np.random.choice(data_indices, val_size, replace=False)
        self.indices = {
            'train': np.array(list(set(data_indices) - set(indices))),
            'val': indices
        }

    def __len__(self):
        return len(self.data['train']) + len(self.data['val'])

    def __str__(self):
        return str(self.data['train'])

    def get(self, batch_size):
        loaders = dict()
        loaders['train'] = DataLoader(self.data['train'], batch_size=batch_size, shuffle=True, pin_memory=True)
        loaders['val'] = DataLoader(self.data['val'], batch_size=batch_size, shuffle=True, pin_memory=True)
        return loaders


class UrbanSound8KLoader(Loader):
    def __init__(self):
        super(UrbanSound8KLoader, self).__init__()
        self.data['train'] = dataset.UrbanReduced(self.indices['train'])
        self.data['val'] = dataset.UrbanReduced(self.indices['val'])


class UrbanMelSpectrogramLoader(Loader):
    def __init__(self):
        super(UrbanMelSpectrogramLoader, self).__init__()
        self.data['train'] = dataset.UrbanMelSpectrogram(self.indices['train'])
        self.data['val'] = dataset.UrbanMelSpectrogram(self.indices['val'])
