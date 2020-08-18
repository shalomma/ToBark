from torch import manual_seed
from torch.utils.data import DataLoader
import dataset


class Loader:
    seed = 14

    def __init__(self):
        manual_seed(self.seed)
        self.data = dict()

    def get(self, batch_size):
        loaders = dict()
        loaders['train'] = DataLoader(self.data['train'], batch_size=batch_size, shuffle=True, pin_memory=True)
        loaders['val'] = DataLoader(self.data['val'], batch_size=batch_size, shuffle=True, pin_memory=True)
        return loaders


class UrbanSound8KLoader(Loader):
    def __init__(self, compose):
        super(UrbanSound8KLoader, self).__init__()
        root_dir = './UrbanSound8K/audio/'
        metadata_file = './UrbanSound8K/metadata/UrbanSound8K.csv'
        self.data['train'] = dataset.UrbanSound8K(metadata_file, root_dir, phase='train', transform=compose)
        self.data['val'] = dataset.UrbanSound8K(metadata_file, root_dir, phase='val', transform=compose)
