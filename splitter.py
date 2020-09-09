import numpy as np
from torch import manual_seed
from torch.utils.data import DataLoader
from dataset import Dataset


class Splitter:
    seed = 14

    def __init__(self, batch_size):
        np.random.seed(self.seed)
        manual_seed(self.seed)
        self.batch_size = batch_size
        self.size = None

    def get(self, dataset: Dataset):
        raise NotImplementedError

    def get_all(self, dataset):
        return DataLoader(dataset(np.arange(0, dataset.size)), batch_size=self.batch_size, shuffle=True)

    @staticmethod
    def print(dataset, indices):
        print(f"{dataset} train (size={len(indices['train'])})")
        print(f"{dataset} val (size={len(indices['val'])})")

    @staticmethod
    def random(size, ratio):
        train_size = int(size * ratio)
        data_indices = np.arange(0, size)
        train_indices = np.random.choice(data_indices, train_size, replace=False)
        return {
            'train': train_indices,
            'val': np.array(list(set(data_indices) - set(train_indices)))
        }


class TestSplitter(Splitter):
    def __init__(self, batch_size, train_ratio=0.9):
        super(TestSplitter, self).__init__(batch_size)
        self.train_ratio = train_ratio

    def get(self, dataset: Dataset):
        indices = self.random(dataset.size, self.train_ratio)
        self.print(dataset, indices)
        return {
            'train': DataLoader(dataset(indices['train']), batch_size=self.batch_size, shuffle=True),
            'val': DataLoader(dataset(indices['val']), batch_size=self.batch_size, shuffle=True)
        }


class KSplitter(Splitter):
    def __init__(self, batch_size, k):
        super(KSplitter, self).__init__(batch_size)
        self.k = k

    def get(self, dataset: Dataset):
        k_splits = []
        for k in range(self.k):
            print(f'Fold {k}')
            indices = self.random(dataset.size, 1 - 1 / self.k)
            self.print(dataset, indices)
            k_splits.append({
                'train': DataLoader(dataset(indices['train']), batch_size=self.batch_size, shuffle=True),
                'val': DataLoader(dataset(indices['val']), batch_size=self.batch_size, shuffle=True)
            })
        return k_splits
