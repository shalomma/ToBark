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
    loaders = loader.MelSpecEncodedLoader(prefix='catsanddogs', size=277, train_ratio=1).get(277)
    data = next(iter(loaders['train']))
    x, y = data['wave'], data['class']

    y_hat = model(x).argmax(axis=1)
    y = y.detach().numpy()
    y_hat = y_hat.detach().numpy()
    y = y == 3
    y_hat = y_hat == 3

    print('Recall: ', metrics.recall_score(y, y_hat))
    print('Precision: ', metrics.precision_score(y, y_hat))
