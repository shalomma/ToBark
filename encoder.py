import torch
import loader


class Encoder:
    def __init__(self, loaders):
        self.dataset_name = str(loaders).lower()
        self.loaders = loaders.get(1)
        self.features = []
        self.labels = []

    def encode(self):
        features = []
        labels = []
        for phase in ['train', 'val']:
            for data in self.loaders[phase]:
                features.append(data['wave'])
                labels.append(data['class'])
        self.features = torch.cat(features)
        self.labels = torch.cat(labels)

    def dump(self):
        with open(f'{self.dataset_name}_features.pt', 'wb') as f:
            torch.save(self.features, f)
        with open(f'{self.dataset_name}_labels.pt', 'wb') as f:
            torch.save(self.labels, f)


if __name__ == '__main__':
    encoder = Encoder(loader.UrbanSound8KLoader())
    encoder.encode()
    encoder.dump()
