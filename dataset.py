import os
import pandas as pd
import numpy as np
import librosa
import torch
import torch.utils.data as data


class UrbanSound8K(data.Dataset):
    def __init__(self, metadata_file, root_dir, train=True, transform=None):
        df = pd.read_csv(metadata_file)
        mask_val = df['fold'] == 10
        self.metadata = df[~mask_val] if train else df[mask_val]
        self.y = torch.tensor(self.metadata['classID'].values)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.metadata.iloc[idx]
        file_name = os.path.join(os.path.abspath(self.root_dir), 'fold' + str(row['fold']) + '/',
                                 row['slice_file_name'])
        x, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mels = np.mean(librosa.feature.melspectrogram(y=x, sr=sample_rate).T, axis=0)
        mels = torch.tensor(mels).view(1, 16, 8)

        return {
            'wave': mels,
            'class': self.y[idx]
        }


class UrbanEmbedded(data.Dataset):
    def __init__(self, root_dir, indices):
        self.data = torch.load(os.path.join(root_dir, 'data.pt'))[:, :6, :]
        self.y = torch.load(os.path.join(root_dir, 'labels.pt'))
        self.data = self.data[indices]
        self.y = self.y[indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {
            'wave': self.data[idx],
            'class': self.y[idx]
        }


class UrbanMelSpectrogram(data.Dataset):
    def __init__(self, root_dir, indices):
        self.data = torch.load(os.path.join(root_dir, 'mel_data.pt')).view(-1, 1, 16, 8)
        # self.y = torch.load(os.path.join(root_dir, 'mel_labels.pt')) ## TODO: create new mel_labels.py
        metadata_file = './UrbanSound8K/metadata/UrbanSound8K.csv'
        df = pd.read_csv(metadata_file)
        self.y = torch.tensor(df['classID'].values)
        self.data = self.data[indices]
        self.y = self.y[indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {
            'wave': self.data[idx],
            'class': self.y[idx]
        }


if __name__ == '__main__':
    path_ = 'UrbanSound8K/metadata/UrbanSound8K.csv'
    root_ = 'UrbanSound8K'
    data = UrbanSound8K(path_, root_)
    sample_ = next(iter(data))
