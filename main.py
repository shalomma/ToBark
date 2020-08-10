import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.optim import Adam
from trainer import Trainer, TrainConfig
from dataset import UrbanSound8K
from network import Encoder


if __name__ == '__main__':
    root_dir = './UrbanSound8K/audio/'
    metadata_file = './UrbanSound8K/metadata/UrbanSound8K.csv'

    epochs = 100
    batch_size = 256
    learning_rate = 1e-3
    in_channels = 2
    num_hidden = 128
    num_residual_hidden = 32
    num_residual_layers = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = Encoder(in_channels, num_hidden, num_residual_layers, num_residual_hidden)
    optimizer = Adam(params=model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    data = dict()
    data['train'] = UrbanSound8K(metadata_file, root_dir, transform=torchaudio.transforms.Spectrogram())
    data['val'] = UrbanSound8K(metadata_file, root_dir, transform=torchaudio.transforms.Spectrogram())
    loaders = dict()
    loaders['train'] = DataLoader(data['train'], batch_size=batch_size, shuffle=True, pin_memory=True)
    loaders['val'] = DataLoader(data['val'], batch_size=batch_size, shuffle=True, pin_memory=True)

    config = TrainConfig(model, loaders, criterion, optimizer)
    trainer = Trainer(config)
    trainer.n_epochs = epochs
    trainer.train()
