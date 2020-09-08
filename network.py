from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F


seed = 14


class Residual(nn.Module, ABC):
    def __init__(self, channels, num_residual_hidden):
        super(Residual, self).__init__()
        torch.manual_seed(seed)
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=channels,
                      out_channels=num_residual_hidden,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hidden,
                      out_channels=channels,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module, ABC):
    def __init__(self, channels, num_residual_layers, num_residual_hidden):
        super(ResidualStack, self).__init__()
        torch.manual_seed(seed)
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(channels, num_residual_hidden)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class ConvPool2d(nn.Module, ABC):
    def __init__(self, in_channels, out_channels, kernel_size, p):
        super(ConvPool2d, self).__init__()
        torch.manual_seed(seed)
        self._block = nn.Sequential(
            # nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=0),
            nn.Dropout(p)
        )

    def forward(self, x):
        return self._block(x)


class MelCNN2d(nn.Module, ABC):
    def __init__(self, in_channels, n_classes):
        super(MelCNN2d, self).__init__()
        torch.manual_seed(seed)
        channels = [in_channels, 64, 128, 32]
        self._layers = nn.ModuleList([
            ConvPool2d(channels[i], channels[i + 1], 2, 0.02)
            for i in range(len(channels) - 1)
        ])

        self._res_layers = ResidualStack(channels[-1], 2, 16)

        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, n_classes)

    def forward(self, inputs):
        batch, _, _, _ = inputs.shape
        x = inputs
        for layer in self._layers:
            x = layer(x)
        x = self._res_layers(x)
        x = x.view(batch, -1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x
