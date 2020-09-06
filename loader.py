from torch import manual_seed
from torch.utils.data import DataLoader
import dataset
import numpy as np


class Loader:
    seed = 14

    def __init__(self):
        manual_seed(self.seed)
        self.data = dict()

    def __len__(self):
        return len(self.data['train']) + len(self.data['val'])

    def get(self, batch_size):
        loaders = dict()
        loaders['train'] = DataLoader(self.data['train'], batch_size=batch_size, shuffle=True, pin_memory=True)
        loaders['val'] = DataLoader(self.data['val'], batch_size=batch_size, shuffle=True, pin_memory=True)
        return loaders


class UrbanSound8KLoader(Loader):
    def __init__(self):
        super(UrbanSound8KLoader, self).__init__()
        root_dir = './data/UrbanSound8K/audio/'
        metadata_file = './data/UrbanSound8K/metadata/UrbanSound8K.csv'
        self.data['train'] = dataset.UrbanSound8K(metadata_file, root_dir, train=True)
        self.data['val'] = dataset.UrbanSound8K(metadata_file, root_dir, train=False)


class UrbanMelSpectrogramLoader(Loader):
    def __init__(self):
        super(UrbanMelSpectrogramLoader, self).__init__()
        root_dir = './data/UrbanSound8K/'
        size = 8732
        val_size = 800
        data_indices = np.arange(0, size)
        val_indices = np.random.choice(data_indices, val_size, replace=False)
        train_indices = np.array(list(set(data_indices) - set(val_indices)))
        self.data['train'] = dataset.UrbanMelSpectrogram(root_dir, train_indices)
        self.data['val'] = dataset.UrbanMelSpectrogram(root_dir, val_indices)
