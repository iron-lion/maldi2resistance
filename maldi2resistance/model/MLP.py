from typing import Literal, Optional

import torch
from torch import cuda
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.input = nn.Linear(input_dim, hidden_dim)
        self.layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h_ = self.LeakyReLU(self.input(x))
        h_ = self.LeakyReLU(self.layer_1(h_))
        h_ = self.LeakyReLU(self.layer_2(h_))

        return h_


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.input = nn.Linear(latent_dim, hidden_dim)
        self.layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h_ = self.LeakyReLU(self.input(x))
        h_ = self.LeakyReLU(self.layer_1(h_))

        output = torch.sigmoid(self.layer_2(h_))
        return output


class AeBasedMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=4096,
        latent_dim=2048,
        device: Optional[Literal["cuda", "cpu"]] = None,
    ):
        super(AeBasedMLP, self).__init__()

        self.encoder = Encoder(
            input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim
        )
        self.decoder = Decoder(
            latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=output_dim
        )

        if device is None:
            if cuda.is_available() and cuda.device_count() > 0:
                device = "cuda"
            else:
                device = "cpu"

        self.to(torch.device(device))

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)

        return output
