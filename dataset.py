import os
import pandas as pd
import numpy as np
import librosa
import torch
import torch.utils.data as data


class UrbanSound8K(data.Dataset):
    def __init__(self):
        self.root_dir = './data/UrbanSound8K/'
        self.metadata_file = os.path.join(self.root_dir, 'metadata/UrbanSound8K.csv')
        self.data = None
        self.y = None

    def __str__(self):
        return str(self.__class__.__name__)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {
            'wave': self.data[idx],
            'class': self.y[idx]
        }


class UrbanReduced(UrbanSound8K):
    def __init__(self, indices):
        super(UrbanReduced, self).__init__()
        df = pd.read_csv(self.metadata_file)
        self.data = df.iloc[indices]
        self.y = torch.tensor(self.data['classID'].values)
        self.root_dir = os.path.join(self.root_dir, 'audio')

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]
        file_name = os.path.join(self.root_dir, 'fold' + str(row['fold']) + '/', row['slice_file_name'])
        x, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mel = np.mean(librosa.feature.melspectrogram(y=x, sr=sample_rate).T, axis=0)
        mel = torch.tensor(mel).view(1, 16, 8)

        return {
            'wave': mel,
            'class': self.y[idx]
        }


class UrbanEmbedded(UrbanSound8K):
    def __init__(self, indices):
        super(UrbanEmbedded, self).__init__()
        self.data = torch.load(os.path.join(self.root_dir, 'data.pt'))[:, :6, :]
        self.y = torch.load(os.path.join(self.root_dir, 'labels.pt'))
        self.data = self.data[indices]
        self.y = self.y[indices]


class UrbanMelSpectrogram(UrbanSound8K):
    def __init__(self, indices):
        super(UrbanMelSpectrogram, self).__init__()
        self.data = torch.load(os.path.join(self.root_dir, 'mel_data.pt')).view(-1, 1, 16, 8)
        # self.y = torch.load(os.path.join(self.root_dir, 'mel_labels.pt')) ## TODO: create new mel_labels.py
        df = pd.read_csv(self.metadata_file)
        self.y = torch.tensor(df['classID'].values)
        self.data = self.data[indices]
        self.y = self.y[indices]


if __name__ == '__main__':
    data = UrbanSound8K()
    sample_ = next(iter(data))
