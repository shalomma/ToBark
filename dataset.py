import os
import pandas as pd
import torchaudio
import torch
import torch.utils.data as data


class UrbanSound8K(data.Dataset):
    """UrbanSound8K dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with the audio files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.metadata.iloc[idx]
        file_name = os.path.join(os.path.abspath(self.root_dir), 'fold' + str(row["fold"]) + '/',
                                 str(row["slice_file_name"]))
        class_name = row["class"]
        wave, sample_rate = torchaudio.load(file_name)
        sample = {'wave': wave, 'class': class_name}

        if self.transform:
            sample = self.transform(sample)

        return sample
