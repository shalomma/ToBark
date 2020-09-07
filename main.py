import torch
from torch.optim import Adam
from torchsummary import summary

import loader as ld
from trainer import Trainer, TrainConfig, TrainCache
import network


if __name__ == '__main__':
    params = {
        'epochs': 1200,
        'batch_size': 2048,
        'learning_rate': 1e-3,
        'in_channels': 6,
        'n_classes': 11
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = network.MelCNN2d(in_channels=params['in_channels'], n_classes=params['n_classes']).to(device)
    summary(model, input_size=(params['in_channels'], 16, 8))
    optimizer = Adam(params=model.parameters(), lr=params['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    loaders = ld.MelSpecEncodedLoader(prefix='UrbanSound8K', size=8732).get(params['batch_size'])

    config = TrainConfig(model, loaders, criterion, optimizer)
    trainer = Trainer(config)
    trainer.n_epochs = params['epochs']
    trainer.train()
    TrainCache().save(model, params)
