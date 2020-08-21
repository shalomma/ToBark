import torch
from sklearn import metrics


class TrainConfig:
    def __init__(self, model, loaders, criterion, optimizer):
        self.model = model
        self.fetchers = loaders
        self.criterion = criterion
        self.optimizer = optimizer


class Trainer:
    def __init__(self, config):
        self.config = config
        self.n_epochs = 50
        self.fold = 0
        self.phases = ['train', 'val']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        for i in range(self.n_epochs):
            to_print = f'Batch {i:04}: '
            for phase in self.phases:
                self.config.model.train() if phase == 'train' else self.config.model.eval()
                data = next(iter(self.config.fetchers[phase]))
                inputs, labels = data['wave'], data['class'].float()
                self.config.optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = self.config.model(inputs).view(-1)

                    loss = self.config.criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        self.config.optimizer.step()

                labels = labels.detach().cpu().numpy()
                # pred = outputs.argmax(axis=1).detach().cpu().numpy()
                pred = (outputs > 0.5).detach().cpu().numpy()
                recall = metrics.recall_score(labels, pred)
                precision = metrics.precision_score(labels, pred)
                f1 = metrics.f1_score(labels, pred)
                # acc = torch.eq(labels, outputs.argmax(axis=1)).numpy().mean()
                to_print += f'{phase} loss: {loss.item():.4f} recall: {recall:.3f} ' \
                            f'precision: {precision:.3f} f1: {f1:.3f}\t'
            print(to_print)
