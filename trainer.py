import os
import copy
import torch
import pickle
import inspect
from datetime import datetime
from git import Repo
from torch.backends import cudnn
from sklearn import metrics

cudnn.deterministic = True
cudnn.benchmark = False
cudnn.fastest = True


class TrainConfig:
    def __init__(self, model, loaders, criterion, optimizer, scheduler=None):
        self.model = model
        self.loaders = loaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler


class Trainer:
    def __init__(self, config):
        self.config = config
        self.n_epochs = 50
        self.fold = 0
        self.phases = ['train', 'val']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos_class = 1

    def train(self):
        best_metric = 0
        best_epoch = 0
        best_state = None
        for i in range(self.n_epochs):
            to_print = f'Epoch {i:04}: '
            for phase in self.phases:
                running_loss = 0.0
                running_labels = torch.tensor([]).to(self.device)
                running_outputs = torch.tensor([]).to(self.device)
                self.config.model.train() if phase == 'train' else self.config.model.eval()
                for data in self.config.loaders[phase]:
                    inputs, labels = data['x'], data['y']
                    self.config.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.config.model(inputs)

                        loss = self.config.criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            self.config.optimizer.step()
                            if self.config.scheduler is not None:
                                self.config.scheduler.step()

                    running_loss += loss.item()
                    running_labels = torch.cat((running_labels, labels))
                    running_outputs = torch.cat((running_outputs, outputs))

                acc = torch.eq(running_labels, running_outputs.argmax(dim=1)).cpu().numpy().mean()
                loss = running_loss / len(self.config.loaders[phase].dataset)
                to_print += f'{phase} loss: {loss:.4f} acc: {acc:.4f}\t'
                if phase == 'val':
                    running_labels = running_labels.cpu().numpy() == self.pos_class
                    running_outputs = running_outputs.argmax(dim=1).cpu().numpy() == self.pos_class
                    recall = metrics.recall_score(running_labels, running_outputs)
                    precision = metrics.precision_score(running_labels, running_outputs)
                    f1 = metrics.f1_score(running_labels, running_outputs)
                    to_print += f'f1: {f1:.4f} recall: {recall:.4f} precision: {precision:.4f}\t'
                    if f1 > best_metric:
                        best_epoch = i
                        best_metric = f1
                        best_state = copy.deepcopy(self.config.model.state_dict())
            print(to_print)
        self.config.model.load_state_dict(best_state)
        print(f'Best epoch: {best_epoch} ({best_metric:.4f})')


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
