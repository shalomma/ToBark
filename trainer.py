import torch


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
        for epoch in range(self.n_epochs):
            to_print = ''
            for phase in self.phases:
                running_loss = 0.0
                self.config.model.train() if phase == 'train' else self.config.model.eval()
                # for data in self.config.fetchers[phase]:
                data = next(iter(self.config.fetchers[phase]))
                inputs, labels = data['wave'], data['class']
                self.config.optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = self.config.model(inputs)

                    loss = self.config.criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        self.config.optimizer.step()

                running_loss += loss.item()
                to_print += f'{phase}: {running_loss:.4f}\t'
            print(to_print)
