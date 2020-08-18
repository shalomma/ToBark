import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, num_hidden, num_residual_hidden):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hidden,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hidden,
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