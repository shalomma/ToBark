import os
import torch
from tqdm import tqdm

import dataset as ds
from loader import Loader


class Encoder:
    def __init__(self, loader, batch_size):
        self.dataset_name = str(loader)
        self.loader = loader.get_all(batch_size)
        self.features = []
        self.labels = []

    def encode(self):
        features = []
        labels = []
        for data in tqdm(self.loader):
            features.append(data['x'])
            labels.append(data['y'])
        self.features = torch.cat(features)
        self.labels = torch.cat(labels)

    def dump(self):
        if not os.path.exists('data'):
            os.makedirs('data')
        with open(f'data/{self.dataset_name}_features.pt', 'wb') as f:
            torch.save(self.features, f)
        with open(f'data/{self.dataset_name}_labels.pt', 'wb') as f:
            torch.save(self.labels, f)


if __name__ == '__main__':
    encoder = Encoder(Loader(ds.UrbanSound8K()), batch_size=32)
    encoder.encode()
    encoder.dump()
