import os
import pandas as pd
import torchaudio
import torch
import torch.utils.data as data


class UrbanSound8K(data.Dataset):
    def __init__(self, metadata_file, root_dir, phase='train', transform=None):
        df = pd.read_csv(metadata_file)
        mask_val = df['fold'] == 10
        self.metadata = df[~mask_val] if phase == 'train' else df[mask_val]
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


if __name__ == '__main__':
    path = 'UrbanSound8K/metadata/UrbanSound8K.csv'
    root = 'UrbanSound8K/audio'
    data = UrbanSound8K(path, root)
    sample_ = next(iter(data))
