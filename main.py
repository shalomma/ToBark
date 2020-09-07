import torch
import torch.optim as optim
from torchsummary import summary

import loader as ld
from trainer import Trainer, TrainConfig, TrainCache
import network


if __name__ == '__main__':
    params = {
        'epochs': 150,
        'batch_size': 256,
        'learning_rate': 1e-3,
        'in_channels': 1,
        'n_classes': 11
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = network.MelCNN2d(in_channels=params['in_channels'], n_classes=params['n_classes']).to(device)
    summary(model, input_size=(params['in_channels'], 16, 8))
    optimizer = optim.Adam(params=model.parameters(), lr=params['learning_rate'])
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, params['learning_rate'],
                                            10 * params['learning_rate'], cycle_momentum=False)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    loaders = ld.MelSpecEncodedLoader(prefixes=['UrbanSound8K', 'CatsAndDogs'], size=8732).get(params['batch_size'])

    config = TrainConfig(model, loaders, criterion, optimizer, scheduler)
    trainer = Trainer(config)
    trainer.n_epochs = params['epochs']
    trainer.train()
    TrainCache().save(model, params)
