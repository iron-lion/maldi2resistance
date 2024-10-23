import torch
from multimodal_amr.models.modules import ResMLP
from torch import nn


class MultilabelResMLP(nn.Module):
    def __init__(self, input_dim, output_dim,hidden_dim = 1024, depth = 5):
        super(MultilabelResMLP, self).__init__()
        self.first = nn.Linear(input_dim, hidden_dim)
        self.res_mlp = ResMLP(
            n_layers=depth,
            dim=hidden_dim,
            output_dim=output_dim,
            p_dropout=0.2,
        )

    def forward(self, x):
        h_ = self.first(x)
        h_ = self.res_mlp(h_)

        output = torch.sigmoid(h_)

        return output