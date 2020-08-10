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
            for phase in self.phases:
                running_loss = 0.0
                running_labels = torch.tensor([]).to(self.device)
                running_outputs = torch.tensor([]).to(self.device)
                if phase == 'train':
                    self.config.model.train()
                else:
                    self.config.model.eval()

                for data in self.config.fetchers[phase]:
                    inputs, labels = data['wave'], data['class']
                    self.config.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.config.model(inputs)

                        loss = self.config.criterion(outputs, labels)
                        if phase == 'train':
                            loss += self.config.l2_regularizer()
                            loss.backward()
                            self.config.optimizer.step()

                    running_loss += loss.item()
                    running_labels = torch.cat((running_labels, labels))
                    running_outputs = torch.cat((running_outputs, outputs))

                loss = running_loss / len(self.config.fetchers[phase])
                print(f'{phase}: {loss:.4f}')
