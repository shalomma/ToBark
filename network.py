from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F


seed = 14


class Residual(nn.Module, ABC):
    def __init__(self, in_channels, num_hidden, num_residual_hidden):
        super(Residual, self).__init__()
        torch.manual_seed(seed)
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=num_residual_hidden,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(in_channels=num_residual_hidden,
                      out_channels=num_hidden,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module, ABC):
    def __init__(self, in_channels, num_hidden, num_residual_layers, num_residual_hidden):
        super(ResidualStack, self).__init__()
        torch.manual_seed(seed)
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hidden, num_residual_hidden)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class MelCNN2d(nn.Module, ABC):
    def __init__(self, in_channels, n_classes):
        super(MelCNN2d, self).__init__()
        torch.manual_seed(seed)
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=64,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._pool_1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self._conv_2 = nn.Conv2d(in_channels=64,
                                 out_channels=128,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._pool_2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self._conv_3 = nn.Conv2d(in_channels=128,
                                 out_channels=64,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._pool_3 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(4160, 1024)
        self.fc2 = nn.Linear(1024, n_classes)

    def forward(self, inputs):
        batch, _, _, _ = inputs.shape
        x = self._conv_1(inputs)
        x = torch.tanh(x)
        x = self._pool_1(x)
        x = self._conv_2(x)
        x = torch.tanh(x)
        x = self._pool_2(x)
        x = self.dropout(x)
        x = self._conv_3(x)
        x = torch.tanh(x)
        x = self._pool_3(x)
        x = x.view(batch, -1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x
