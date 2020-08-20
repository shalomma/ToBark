import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, num_hidden, num_residual_hidden):
        super(Residual, self).__init__()
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


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hidden, num_residual_layers, num_residual_hidden):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hidden, num_residual_hidden)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class WaveCNN(nn.Module):
    def __init__(self, in_channels, num_residual_layers):
        super(WaveCNN, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=1,
                                 kernel_size=3,
                                 stride=2, padding=0)
        self._pool_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        self._conv_2 = nn.Conv2d(in_channels=1,
                                 out_channels=1,
                                 kernel_size=5,
                                 stride=2, padding=0)
        self._pool_2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        self._conv_3 = nn.Conv2d(in_channels=1,
                                 out_channels=1,
                                 kernel_size=5,
                                 stride=2, padding=0)
        self._pool_3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        self._conv_4 = nn.Conv2d(in_channels=1,
                                 out_channels=1,
                                 kernel_size=5,
                                 stride=2, padding=0)
        self._pool_4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        self._conv_5 = nn.Conv2d(in_channels=1,
                                 out_channels=1,
                                 kernel_size=5,
                                 stride=2, padding=0)

        self._residual_stack = ResidualStack(in_channels=1,
                                             num_hidden=1,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hidden=1)
        self.fc = nn.Linear(22, 10)

    def forward(self, inputs):
        batch, h, w = inputs.shape
        x = inputs.view(batch, 1, h, w)
        x = self._conv_1(x)
        x = F.relu(x)
        x = self._pool_1(x)

        x = self._conv_2(x)
        x = F.relu(x)
        x = self._pool_2(x)

        x = self._conv_3(x)
        x = F.relu(x)
        x = self._pool_3(x)

        x = self._conv_4(x)
        x = F.relu(x)
        x = self._pool_4(x)

        x = self._conv_5(x)
        x = self._residual_stack(x)
        x = x.view(batch, -1)
        x = self.fc(x)
        return x


class EmbeddedWaveCNN(nn.Module):
    def __init__(self, in_channels, num_residual_layers):
        super(EmbeddedWaveCNN, self).__init__()

        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=4,
                                 kernel_size=15,
                                 stride=2, padding=0, dilation=2)
        self._pool_1 = nn.MaxPool1d(kernel_size=7, stride=1, padding=0)
        self._conv_2 = nn.Conv1d(in_channels=4,
                                 out_channels=2,
                                 kernel_size=15,
                                 stride=2, padding=0, dilation=2)
        self._pool_2 = nn.MaxPool1d(kernel_size=3, stride=1, padding=0)
        self._conv_3 = nn.Conv1d(in_channels=2,
                                 out_channels=2,
                                 kernel_size=5,
                                 stride=2, padding=0, dilation=2)
        self._pool_3 = nn.MaxPool1d(kernel_size=3, stride=1, padding=0)
        self._residual_stack = ResidualStack(in_channels=2,
                                             num_hidden=1,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hidden=1)
        self._conv_4 = nn.Conv1d(in_channels=2,
                                 out_channels=1,
                                 kernel_size=5,
                                 stride=2, padding=0, dilation=1)
        self.fc = nn.Linear(21, 10)

    def forward(self, inputs):
        batch, _, _ = inputs.shape
        x = inputs
        x = self._conv_1(x)
        x = F.relu(x)
        x = self._pool_1(x)
        x = self._conv_2(x)
        x = F.relu(x)
        x = self._pool_2(x)
        x = self._conv_3(x)
        x = F.relu(x)
        x = self._pool_3(x)
        x = self._residual_stack(x)
        x = self._conv_4(x)
        x = x.view(batch, -1)
        x = self.fc(x)
        return x


class MelCNN2d(nn.Module):
    def __init__(self, in_channels):
        super(MelCNN2d, self).__init__()
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
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(10752, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, inputs):
        batch, _, _, _ = inputs.shape
        x = self._conv_1(inputs)
        x = torch.tanh(x)
        x = self._pool_1(x)
        x = self._conv_2(x)
        x = torch.tanh(x)
        x = self._pool_2(x)
        x = self.dropout(x)
        x = x.view(batch, -1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x

