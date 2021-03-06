import torch
import argparse
import torch.optim as optim
from torchsummary import summary

from dataset import MelSpecEncoded
import splitter as sp
from trainer import Trainer, TrainConfig, TrainCache
import network

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='data epochs', default=150)
    parser.add_argument('--folds', type=int, help='k folds', default=10)
    parser.add_argument('--scheduler', dest='scheduler', help='cyclicLR scheduler', action='store_true')
    parser.add_argument('--binary', dest='binary', help='binary classification', action='store_true')
    args = parser.parse_args()

    params = {
        'epochs': args.epochs,
        'k_folds': args.folds,
        'batch_size': 256,
        'learning_rate': 5e-4,
        'weight_decay': 0.1,
        'in_channels': 1,
        'pos_weight': 2.,
        'n_classes': 2 if args.binary else 10
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = network.MelCNN2d(in_channels=params['in_channels'], n_classes=params['n_classes']).to(device)
    summary(model, input_size=(params['in_channels'], 16, 8))
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., params['pos_weight']]).to(device), reduction='sum')

    prefixes = ['UrbanSound8K', 'ESC50']
    if args.binary:
        prefixes = [p + '_binary' for p in prefixes]
    dataset = MelSpecEncoded(prefixes)
    k_loaders = sp.KSplitter(params['batch_size'], k=args.folds).get(dataset)

    for loaders in k_loaders:
        model = network.MelCNN2d(in_channels=params['in_channels'], n_classes=params['n_classes']).to(device)
        optimizer = optim.Adam(params=model.parameters(), lr=params['learning_rate'],
                               weight_decay=params['weight_decay'])
        scheduler = \
            optim.lr_scheduler.CyclicLR(optimizer, params['learning_rate'],
                                        10 * params['learning_rate'], cycle_momentum=False) \
            if args.scheduler else None
        config = TrainConfig(model, loaders, criterion, optimizer, scheduler)
        trainer = Trainer(config)
        trainer.n_epochs = params['epochs']
        trainer.train()
        TrainCache().save(model, params)
