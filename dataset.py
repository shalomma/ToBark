import os
import glob
import torch
import librosa
import pandas as pd
import numpy as np
import torch.utils.data as data
from abc import ABC, abstractmethod
import scipy.stats as stats


class Dataset(data.Dataset, ABC):
    def __init__(self):
        self.root_dir = None
        self.metadata = None
        self.y = None

    def __str__(self):
        return str(self.__class__.__name__)

    def __len__(self):
        return len(self.metadata)

    @abstractmethod
    def next_file_name(self, idx):
        pass

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.next_file_name(idx)
        x, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mel = librosa.feature.melspectrogram(y=x, sr=sample_rate).T
        stack = np.array([
            np.mean(mel, axis=0),
            np.median(mel, axis=0),
            np.min(mel, axis=0),
            np.max(mel, axis=0),
            stats.skew(mel, axis=0),
            stats.kurtosis(mel, axis=0)
        ])
        x = torch.tensor(stack).view(-1, 16, 8)

        return {
            'wave': x,
            'class': self.y[idx]
        }


class UrbanSound8K(Dataset):
    size = 8732

    def __init__(self, indices):
        super(UrbanSound8K, self).__init__()
        self.root_dir = './data/UrbanSound8K/'
        self.metadata_file = os.path.join(self.root_dir, 'metadata/UrbanSound8K.csv')
        df = pd.read_csv(self.metadata_file)
        self.metadata = df.iloc[indices]
        self.y = torch.tensor(self.metadata['classID'].values)
        self.root_dir = os.path.join(self.root_dir, 'audio')

    def next_file_name(self, idx):
        row = self.metadata.iloc[idx]
        return os.path.join(self.root_dir, 'fold' + str(row['fold']) + '/', row['slice_file_name'])


class CatsAndDogs(Dataset):
    size = 277

    def __init__(self, indices):
        super(CatsAndDogs, self).__init__()
        self.root_dir = './data/cats_dogs/'
        self.metadata = glob.glob(self.root_dir + '*')
        self.y = [self.parse_class(f) for f in self.metadata]
        self.metadata = list(map(self.metadata.__getitem__, indices))
        self.y = list(map(self.y.__getitem__, indices))

    @staticmethod
    def parse_class(file):
        return 3 if 'dog' in file.split('/')[-1] else 10

    def next_file_name(self, idx):
        return self.metadata[idx]


class MelSpecEncoded:
    def __init__(self, prefix, indices):
        self.prefix = prefix
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        root_dir = './data'
        with open(os.path.join(root_dir, f'{prefix}_features.pt'), 'rb') as f:
            self.data = torch.load(f, map_location=device)
        with open(os.path.join(root_dir, f'{prefix}_labels.pt'), 'rb') as f:
            self.y = torch.load(f, map_location=device)
        self.data = self.data[indices]
        self.y = self.y[indices]

    def __str__(self):
        return str(self.__class__.__name__) + ' ' + self.prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {
            'wave': self.data[idx],
            'class': self.y[idx]
        }
