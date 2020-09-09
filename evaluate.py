import argparse
from sklearn import metrics

import network
from dataset import MelSpecEncoded
import loader as ld
from trainer import TrainCache


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('timestamp', type=str, help='models timestamp')
    args = parser.parse_args()

    model, params = TrainCache.load(network.MelCNN2d, args.timestamp)

    prefixes = ['CatsAndDogs']
    if params['n_classes'] == 2:
        prefixes = [p + '_binary' for p in prefixes]
    dataset = MelSpecEncoded(prefixes)
    loaders = ld.Loader(params['batch_size']).get_all(dataset)

    data = next(iter(loaders))
    x, y = data['x'], data['y']

    y_hat = model(x).argmax(axis=1)
    y = y.detach().cpu().numpy()
    y_hat = y_hat.detach().cpu().numpy()
    y = y == 3
    y_hat = y_hat == 3

    print(f'Accuracy: {metrics.accuracy_score(y, y_hat):.4f}')
    print(f'Recall: {metrics.recall_score(y, y_hat):.4f}')
    print(f'Precision: {metrics.precision_score(y, y_hat):.4f}')
    print(f'Kappa: {metrics.cohen_kappa_score(y, y_hat):.4f}')
    print(f'F1: {metrics.f1_score(y, y_hat):.4f}')
