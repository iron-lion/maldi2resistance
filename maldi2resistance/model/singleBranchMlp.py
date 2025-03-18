import torch
from torch import nn


class SingleBranchMLP(nn.Module):
    def __init__(
        self,
        input_dim=6000,
        output_dim=64,
        layer_dims=[512, 256, 128],
        layer_or_batchnorm="layer",
        dropout=0.2,
    ):
        super().__init__()

        c = input_dim
        layers = []
        for i in layer_dims:
            layers.append(nn.Linear(c, i))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            layers.append(
                nn.LayerNorm(i) if layer_or_batchnorm == "layer" else nn.BatchNorm1d(i)
            )
            c = i

        layers.append(nn.Linear(c, output_dim))

        self.net = nn.Sequential(*layers)

        self.hsize = output_dim

    def forward(self, x):
        return torch.sigmoid(self.net(x))