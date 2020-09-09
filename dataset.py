import os
import glob
import torch
import librosa
import pandas as pd
import numpy as np
import torch.utils.data as data
from abc import ABC, abstractmethod


class Dataset(data.Dataset, ABC):
    size = NotImplemented
    root_dir = NotImplemented
    pos_class = None

    @abstractmethod
    def __init__(self):
        self.metadata = None
        self.y = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.str = str(self.__class__.__name__)

    @abstractmethod
    def __call__(self, indices):
        pass

    def __str__(self):
        return self.str

    def __len__(self):
        return len(self.metadata)

    @abstractmethod
    def __add__(self, other):
        pass

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.next_file_name(idx)
        wave, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mel = librosa.feature.melspectrogram(y=wave, sr=sample_rate)
        mel = np.mean(mel, axis=1)
        x = torch.tensor(mel).view(1, 16, 8).to(self.device)

        return {
            'x': x,
            'y': self.y[idx]
        }

    @abstractmethod
    def next_file_name(self, idx):
        pass

    def binarize(self):
        self.y = np.where(self.y == self.pos_class, 1, 0)
        self.str += '_binary'


class UrbanSound8K(Dataset):
    size = 8732
    root_dir = './data/UrbanSound8K/'
    pos_class = 3

    def __init__(self):
        super(UrbanSound8K, self).__init__()
        metadata_file = os.path.join(self.root_dir, 'metadata/UrbanSound8K.csv')
        self.metadata = pd.read_csv(metadata_file)
        self.y = torch.tensor(self.metadata['classID'].values).to(self.device)

    def __call__(self, indices):
        self.metadata = self.metadata.iloc[indices]
        self.y = self.y[indices]
        return self

    def __add__(self, other):
        obj = UrbanSound8K()
        obj.metadata = pd.concat([self.metadata, other.metadata])
        obj.y = torch.cat((self.y, other.y))
        return obj

    def next_file_name(self, idx):
        row = self.metadata.iloc[idx]
        audio_dir = os.path.join(self.root_dir, 'audio')
        return os.path.join(audio_dir, 'fold' + str(row['fold']) + '/', row['slice_file_name'])


class CatsAndDogs(Dataset):
    size = 277
    root_dir = './data/cats_dogs/'
    pos_class = 1

    def __init__(self):
        super(CatsAndDogs, self).__init__()
        self.metadata = glob.glob(self.root_dir + '*')
        self.y = torch.tensor([self.parse_class(f) for f in self.metadata]).to(self.device)

    def __call__(self, indices):
        self.metadata = [self.metadata[i] for i in indices]
        self.y = self.y[indices]
        return self

    def __add__(self, other):
        obj = CatsAndDogs()
        obj.metadata = self.metadata + other.metadata
        obj.y = self.y + other.y
        return obj

    @staticmethod
    def parse_class(file):
        return 1 if 'dog' in file.split('/')[-1] else 0

    def next_file_name(self, idx):
        return self.metadata[idx]


class ESC50(Dataset):
    size = 2000
    root_dir = './data/ESC-50-master/'
    pos_class = 0

    def __init__(self):
        super(ESC50, self).__init__()
        metadata_file = os.path.join(self.root_dir, 'meta/esc50.csv')
        self.metadata = pd.read_csv(metadata_file)
        self.y = torch.tensor(self.metadata['target'].values).to(self.device)

    def __call__(self, indices):
        self.metadata = self.metadata.iloc[indices]
        self.y = self.y[indices]
        return self

    def __add__(self, other):
        obj = ESC50()
        obj.metadata = pd.concat([self.metadata, other.metadata])
        obj.y = torch.cat((self.y, other.y))
        return obj

    def next_file_name(self, idx):
        row = self.metadata.iloc[idx]
        audio_dir = os.path.join(self.root_dir, 'audio')
        return os.path.join(audio_dir, row['filename'])


class IteratedDataset(data.Dataset):
    def __init__(self, x, y):
        self.data = x
        self.y = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {
            'x': self.data[idx],
            'y': self.y[idx]
        }


class MelSpecEncoded(Dataset):
    size = None
    root_dir = './data'

    def __init__(self, prefixes):
        super(MelSpecEncoded, self).__init__()
        self.data = torch.tensor([]).to('cpu')
        self.y = torch.tensor([], dtype=torch.long).to('cpu')
        for prefix in prefixes:
            with open(os.path.join(self.root_dir, f'{prefix}_features.pt'), 'rb') as f:
                self.data = torch.cat((self.data, torch.load(f, map_location='cpu')))
            with open(os.path.join(self.root_dir, f'{prefix}_labels.pt'), 'rb') as f:
                self.y = torch.cat((self.y, torch.load(f, map_location='cpu')))
        self.data = self.data.to(self.device)
        self.y = self.y.to(self.device)
        self.size = len(self.y)

    def __call__(self, indices):
        return IteratedDataset(self.data[indices], self.y[indices])

    def __str__(self):
        return str(self.__class__.__name__)

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        obj = MelSpecEncoded([])
        obj.data = torch.cat((self.data, other.data))
        obj.y = torch.cat((self.y, other.y))
        return obj

    def next_file_name(self, idx):
        pass

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {
            'x': self.data[idx],
            'y': self.y[idx]
        }
