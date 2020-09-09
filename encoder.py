import os
import torch
import pickle
from tqdm import tqdm

import dataset as ds
from loader import Loader


class Encoder:
    def __init__(self, dataset: ds.Dataset):
        self.loader = Loader(batch_size=32).get_all(dataset)
        self.dataset_name = str(dataset)
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
        meta = {
            'size': len(self.labels),
            'n_classes': len(torch.unique(self.labels))
        }
        with open(f'data/{self.dataset_name}_meta.pt', 'wb') as f:
            pickle.dump(meta, f)


if __name__ == '__main__':
    encoder = Encoder(ds.UrbanSound8K())
    encoder.encode()
    encoder.dump()
