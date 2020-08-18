import os
import pandas as pd
import torchaudio
import torch
import torch.utils.data as data


class UrbanSound8K(data.Dataset):
    def __init__(self, metadata_file, root_dir, transform=None):
        self.metadata = pd.read_csv(metadata_file)
        self.y = torch.tensor(self.metadata['classID'])
        self.root_dir = root_dir
        self.transform = transform
        self.max_frames = 176400

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.metadata.iloc[idx]
        file_name = os.path.join(os.path.abspath(self.root_dir), 'fold' + str(row['fold']) + '/',
                                 row['slice_file_name'])
        wave, sample_rate = torchaudio.load(file_name)
        wave = wave[0]
        length = len(wave)
        if length % 2 == 1:
            wave = wave[:-1]
        if length > self.max_frames:
            mid_idx = length // 2
            wave = wave[mid_idx - self.max_frames // 2:mid_idx + self.max_frames // 2]
        elif length < self.max_frames:
            mid_idx = self.max_frames // 2
            t = torch.zeros(self.max_frames)
            t[mid_idx - length // 2:mid_idx + length // 2] = t[mid_idx - length // 2:mid_idx + length // 2] + wave
            wave = t

        if self.transform:
            wave = self.transform(wave)

        return {
            'wave': wave,
            'class': self.y[idx]
        }


if __name__ == '__main__':
    path = 'UrbanSound8K/metadata/UrbanSound8K.csv'
    root = 'UrbanSound8K/audio'
    data = UrbanSound8K(path, root)
    sample = next(iter(data))
