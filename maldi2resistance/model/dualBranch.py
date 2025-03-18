from torch import nn


class DualBranchOneHot(nn.Module):
    def __init__(
            self,
            input_dim_spectrum,
            input_dim_drug,
            output_dim = 64,
            layer_dims=[512, 256, 128],
            layer_or_batchnorm="layer",
            dropout=0.2,
    ):
        super().__init__()

        c = input_dim_spectrum
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

        self.spectrum_embedder = nn.Sequential(*layers)
        self.drug_embedder = nn.Embedding(input_dim_drug, 64)

        self.hsize = output_dim
        self.scale = True

    def forward(self, x, drug):
        drug_embedding = self.drug_embedder(drug)
        spectrum_embedding = self.spectrum_embedder(x)
        norm = 1 if not self.scale else spectrum_embedding.shape[-1] ** 0.5
        return (drug_embedding * spectrum_embedding).sum(-1) / norm
