import os
import torch
import pickle
import inspect
from datetime import datetime
from git import Repo


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
                inputs, labels = data['wave'], data['class']
                self.config.optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = self.config.model(inputs)

                    loss = self.config.criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        self.config.optimizer.step()

                acc = torch.eq(labels, outputs.argmax(axis=1)).numpy().mean()
                to_print += f'{phase} loss: {loss.item():.4f} acc: {acc:.4f}\t'
            print(to_print)


class TrainCache:
    def __init__(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        timestamp = str(datetime.now())[:-7]
        timestamp = timestamp.replace('-', '_').replace(' ', '_').replace(':', '_')
        self.directory = f'models/{timestamp}'
        print(self.directory)
        os.makedirs(self.directory)

    def save(self, model, params):
        params['commit'] = Repo('./').head.commit.hexsha[:7]
        with open(f'{self.directory}/params.pkl', 'wb') as f:
            pickle.dump(params, f)
        with open(f'{self.directory}/model.pt', 'wb') as f:
            torch.save(model.state_dict(), f)

    @classmethod
    def load(cls, model_class, timestamp):
        with open(f'models/{timestamp}/params.pkl', 'rb') as f:
            params = pickle.load(f)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args = list(inspect.signature(model_class.__init__).parameters.keys())
        params_class = dict()
        for a in args[1:]:
            params_class[a] = params[a]
        model = model_class(**params_class).to(device)
        with open(f'models/{timestamp}/model.pt', 'rb') as f:
            state_dict = torch.load(f, map_location=device)
            model.load_state_dict(state_dict)
        model.eval()
        return model, params
