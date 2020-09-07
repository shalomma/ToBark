import argparse
from sklearn import metrics

import network
import loader
from trainer import TrainCache


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('timestamp', type=str, help='models timestamp')
    args = parser.parse_args()

    model, params = TrainCache.load(network.MelCNN2d, args.timestamp)
    loaders = loader.MelSpecEncodedLoader(prefix='CatsAndDogs', size=277, train_ratio=0.5).get_all(277)
    data = next(iter(loaders))
    x, y = data['x'], data['y']

    y_hat = model(x).argmax(axis=1)
    y = y.detach().cpu().numpy()
    y_hat = y_hat.detach().cpu().numpy()
    y = y == 3
    y_hat = y_hat == 3

    print('Accuracy: ', metrics.accuracy_score(y, y_hat))
    print('Recall: ', metrics.recall_score(y, y_hat))
    print('Precision: ', metrics.precision_score(y, y_hat))
    print('Kappa: ', metrics.cohen_kappa_score(y, y_hat))
    print('F1: ', metrics.f1_score(y, y_hat))
