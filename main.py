import torch
from torch.optim import Adam
from torchvision.transforms import Compose

import loader
from trainer import Trainer, TrainConfig
from network import WaveCNN
import transforms


if __name__ == '__main__':
    epochs = 1000
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
        transforms.CutAndPaste(max_frames),
        transforms.Spectrogram()
    ])
    loaders = loader.UrbanSound8KLoader(compose).get(batch_size)

    config = TrainConfig(model, loaders, criterion, optimizer)
    trainer = Trainer(config)
    trainer.n_epochs = epochs
    trainer.train()
