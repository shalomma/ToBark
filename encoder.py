import os
import torch
import loader as ld
import time
import datetime


class Encoder:
    def __init__(self, loader, batch_size):
        self.dataset_name = str(loader)
        self.loader = loader.get_all(batch_size)
        self.features = []
        self.labels = []

    def encode(self):
        features = []
        labels = []
        start = time.time()
        for data in self.loader:
            features.append(data['x'])
            labels.append(data['y'])
        self.features = torch.cat(features)
        self.labels = torch.cat(labels)
        end = time.time() - start
        print(f'Pipeline runtime: {datetime.timedelta(seconds=int(end))}')

    def dump(self):
        if not os.path.exists('data'):
            os.makedirs('data')
        with open(f'data/{self.dataset_name}_features.pt', 'wb') as f:
            torch.save(self.features, f)
        with open(f'data/{self.dataset_name}_labels.pt', 'wb') as f:
            torch.save(self.labels, f)


if __name__ == '__main__':
    encoder = Encoder(ld.UrbanSound8KLoader(), batch_size=32)
    encoder.encode()
    encoder.dump()
