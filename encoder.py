import os
import torch
import pickle
from tqdm import tqdm

import dataset as ds
from splitter import Splitter


class Encoder:
    def __init__(self, dataset: ds.Dataset):
        self.loader = Splitter(batch_size=32).get_all(dataset)
        self.dataset_name = str(dataset)
        self.features = []
        self.labels = []
        self.pos_class = dataset.pos_class
        print(f'Encoding: {self.dataset_name}')

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
            'n_classes': len(torch.unique(self.labels)),
            'pos_class': self.pos_class
        }
        with open(f'data/{self.dataset_name}_meta.pt', 'wb') as f:
            pickle.dump(meta, f)
        print(meta)


if __name__ == '__main__':
    dataset_ = ds.UrbanSound8K()
    dataset_.binarize()
    encoder = Encoder(dataset_)
    encoder.encode()
    encoder.dump()
