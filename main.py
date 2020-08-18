import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.optim import Adam
from trainer import Trainer, TrainConfig
from dataset import UrbanSound8K
from network import WaveCNN
from torchvision.transforms import Compose
from transforms import CutAndPaste


if __name__ == '__main__':
    root_dir = './UrbanSound8K/audio/'
    metadata_file = './UrbanSound8K/metadata/UrbanSound8K.csv'

    epochs = 100
    batch_size = 128
    learning_rate = 1e-3
    in_channels = 1
    num_hidden = 128
    num_residual_hidden = 32
    num_residual_layers = 2

    max_frames = 176400

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = WaveCNN(in_channels, num_residual_layers)
    optimizer = Adam(params=model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    compose = Compose([
        CutAndPaste(max_frames),
        torchaudio.transforms.Spectrogram()
    ])
    data = dict()
    data['train'] = UrbanSound8K(metadata_file, root_dir, transform=compose)
    data['val'] = UrbanSound8K(metadata_file, root_dir, transform=compose)
    loaders = dict()
    loaders['train'] = DataLoader(data['train'], batch_size=batch_size, shuffle=True, pin_memory=True)
    loaders['val'] = DataLoader(data['val'], batch_size=batch_size, shuffle=True, pin_memory=True)

    config = TrainConfig(model, loaders, criterion, optimizer)
    trainer = Trainer(config)
    trainer.n_epochs = epochs
    trainer.train()
