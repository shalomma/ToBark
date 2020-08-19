import os
import pandas as pd
import torchaudio
import torch
import torch.utils.data as data
import numpy as np


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
        sample = torchaudio.backend.sox_backend.load(file_name)

        if self.transform:
            sample = self.transform(sample)

        wave, sample_rate = sample

        return {
            'wave': wave,
            'class': self.y[idx]
        }


class UrbanEmbedded(data.Dataset):
    def __init__(self, root_dir, train=True):
        self.data = torch.load(os.path.join(root_dir, 'data.pt'))[:, :6, :]
        self.y = torch.load(os.path.join(root_dir, 'labels.pt'))
        val_size = 5000 if train else 300
        current_indices = np.arange(0, len(self.y))
        indices = np.random.choice(current_indices, val_size, replace=False)
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
