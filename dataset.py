import os
import glob
import torch
import librosa
import pandas as pd
import numpy as np
import torch.utils.data as data
from abc import ABC, abstractmethod


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
    def __add__(self, other):
        pass

    @abstractmethod
    def next_file_name(self, idx):
        pass

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.next_file_name(idx)
        x, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mel = librosa.feature.melspectrogram(y=x, sr=sample_rate)
        mel = np.mean(mel, axis=1)
        x = torch.tensor(mel).view(1, 16, 8)

        return {
            'x': x,
            'y': self.y[idx]
        }


class UrbanSound8K(Dataset):
    size = 8732

    def __init__(self, indices=None):
        super(UrbanSound8K, self).__init__()
        self.root_dir = './data/UrbanSound8K/'
        self.metadata_file = os.path.join(self.root_dir, 'metadata/UrbanSound8K.csv')
        if indices is not None:
            df = pd.read_csv(self.metadata_file)
            self.metadata = df.iloc[indices]
            self.y = torch.tensor(self.metadata['classID'].values)

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

    def __init__(self, indices=None):
        super(CatsAndDogs, self).__init__()
        self.root_dir = './data/cats_dogs/'
        if indices is not None:
            self.metadata = glob.glob(self.root_dir + '*')
            self.y = [self.parse_class(f) for f in self.metadata]
            self.metadata = [self.metadata[i] for i in indices]
            self.y = torch.tensor([self.y[i] for i in indices])

    def __add__(self, other):
        obj = CatsAndDogs()
        obj.metadata = self.metadata + other.metadata
        obj.y = self.y + other.y
        return obj

    @staticmethod
    def parse_class(file):
        return 3 if 'dog' in file.split('/')[-1] else 10

    def next_file_name(self, idx):
        return self.metadata[idx]


class ESC50(Dataset):
    size = 2000

    def __init__(self, indices=None):
        super(ESC50, self).__init__()
        self.root_dir = './data/ESC-50-master/'
        self.metadata_file = os.path.join(self.root_dir, 'meta/esc50.csv')
        if indices is not None:
            df = pd.read_csv(self.metadata_file)
            self.metadata = df.iloc[indices]
            self.y = torch.tensor(self.metadata['target'].values)

    def __add__(self, other):
        obj = ESC50()
        obj.metadata = pd.concat([self.metadata, other.metadata])
        obj.y = torch.cat((self.y, other.y))
        return obj

    def next_file_name(self, idx):
        row = self.metadata.iloc[idx]
        audio_dir = os.path.join(self.root_dir, 'audio')
        return os.path.join(audio_dir, row['filename'])


class MelSpecEncoded:
    def __init__(self, prefixes, indices=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        root_dir = './data'
        if indices is not None:
            self.data = torch.tensor([]).to(device)
            self.y = torch.tensor([], dtype=torch.long).to(device)
            for prefix in prefixes:
                with open(os.path.join(root_dir, f'{prefix}_features.pt'), 'rb') as f:
                    self.data = torch.cat((self.data, torch.load(f, map_location=device)))
                with open(os.path.join(root_dir, f'{prefix}_labels.pt'), 'rb') as f:
                    self.y = torch.cat((self.y, torch.load(f, map_location=device)))
            self.data = self.data[indices]
            self.y = self.y[indices]

    def __str__(self):
        return str(self.__class__.__name__)

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        obj = MelSpecEncoded(None)
        obj.data = torch.cat((self.data, other.data))
        obj.y = torch.cat((self.y, other.y))
        return obj

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {
            'x': self.data[idx],
            'y': self.y[idx]
        }
