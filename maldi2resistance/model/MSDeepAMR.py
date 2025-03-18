"""
This file contains the definition for a customized version of the MSDeepAMR model. It has been modified in such a
way that instead of a single output, it can output a vector, which then corresponds to a prediction for
individual antibiotics and thus solves a multi-label problem.

The original implementation for TensorFlow can be found in this repository:
https://github.com/xlopez-ml/DL-AMR

The corresponding paper is the following:
'MSDeepAMR: antimicrobial resistance prediction based on deep neural networks and transfer learning'
https://doi.org/10.3389/fmicb.2024.1361795
"""

import torch
from torch import nn


class MSDeepAMR(nn.Module):
    def __init__(self, output_dim):
        super(MSDeepAMR, self).__init__()

        self.one = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=17)
        self.one_batch = nn.BatchNorm1d(64)

        self.two = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=9)
        self.two_batch = nn.BatchNorm1d(128)

        self.three = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5)
        self.three_batch = nn.BatchNorm1d(256)

        self.four = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5)
        self.four_batch = nn.BatchNorm1d(256)

        self.flatten = nn.Flatten()

        self.layer_1 = nn.Linear(286720, 256)
        self.layer_2 = nn.Linear(256, 64)
        self.layer_3 = nn.Linear(64, 64)
        self.layer_4 = nn.Linear(64, output_dim)

        self.maxPooling = nn.MaxPool1d(2)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        _h = self.maxPooling(self.LeakyReLU(self.one(x)))
        _h = self.one_batch(_h)

        _h = self.maxPooling(self.LeakyReLU(self.two(_h)))
        _h = self.two_batch(_h)

        _h = self.maxPooling(self.LeakyReLU(self.three(_h)))
        _h = self.three_batch(_h)

        _h = self.maxPooling(self.LeakyReLU(self.four(_h)))
        _h = self.four_batch(_h)

        _h = self.flatten(_h)

        _h = self.LeakyReLU(self.layer_1(_h))
        _h = self.LeakyReLU(self.layer_2(_h))
        _h = self.LeakyReLU(self.layer_3(_h))
        _h = torch.sigmoid(self.layer_4(_h))

        return _h
