import os
import torch
from torch.optim import Adam
from torchsummary import summary
from datetime import datetime

import loader
from trainer import Trainer, TrainConfig
import network
from utils import save_model


if __name__ == '__main__':
    if not os.path.exists('models'):
        os.makedirs('models')
    timestamp = str(datetime.now())[:-7]
    timestamp = timestamp.replace('-', '_').replace(' ', '_').replace(':', '_')
    directory = f'models/{timestamp}'
    os.makedirs(directory)

    params = {
        'epochs': 100,
        'batch_size': 2048,
        'learning_rate': 1e-3,
        'in_channels': 1
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = network.MelCNN2d(in_channels=params['in_channels'])
    summary(model, input_size=(params['in_channels'], 16, 8))
    optimizer = Adam(params=model.parameters(), lr=params['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    loaders = loader.MelSpecEncodedLoader(prefix='urbansound8k', size=8732).get(params['batch_size'])

    config = TrainConfig(model, loaders, criterion, optimizer)
    trainer = Trainer(config)
    trainer.n_epochs = params['epochs']
    trainer.train()
    save_model(model, params, directory)
