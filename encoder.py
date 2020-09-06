import torch
import loader
import time
import datetime


class Encoder:
    def __init__(self, loaders, batch_size):
        self.dataset_name = str(loaders).lower()
        self.loaders = loaders.get(batch_size)
        self.features = []
        self.labels = []

    def encode(self):
        features = []
        labels = []
        start = time.time()
        for phase in ['train', 'val']:
            print(phase)
            for data in self.loaders[phase]:
                features.append(data['wave'])
                labels.append(data['class'])
        self.features = torch.cat(features)
        self.labels = torch.cat(labels)
        end = time.time() - start
        print(f'Pipeline runtime: {datetime.timedelta(seconds=int(end))}')

    def dump(self):
        with open(f'{self.dataset_name}_features.pt', 'wb') as f:
            torch.save(self.features, f)
        with open(f'{self.dataset_name}_labels.pt', 'wb') as f:
            torch.save(self.labels, f)


if __name__ == '__main__':
    encoder = Encoder(loader.UrbanSound8KLoader(), 32)
    encoder.encode()
    encoder.dump()
