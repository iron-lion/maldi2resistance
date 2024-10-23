from pathlib import Path

import numpy as np
import torch
from maldi_learn.vectorization import BinningVectorizer

from src.abstract_model import Model
from src.ms_data import MSData, Resistance, Resistances


class ae(Model):
    def __init__(self):
        super(ae, self).__init__(
            name= "ae_based"
        )
        path = Path("/home/jan/Uni/master/data/model/ae-based.pt")

        self.model = torch.jit.load(path, map_location="cpu")
        self.model.eval()

    def predict(self, data:MSData) -> Resistances:

        transform = BinningVectorizer(18000, min_bin=2000, max_bin=20000)
        min_range = min(data.xValues)
        min_range = min(min_range, transform.min_bin)
        max_range = max(data.xValues)
        max_range = max(max_range, transform.max_bin)
        bin_edges_ = np.linspace(min_range, max_range, transform.n_bins + 1)

        times = data.xValues
        valid = (times > bin_edges_[0]) & (times <= bin_edges_[-1])
        vec = np.histogram(data.xValues, bins=bin_edges_, weights=data.yValues)[0]
        tensor = torch.from_numpy(np.float32(vec))

        latent, output = self.model(tensor)

        antibiotics = ['Penicillin', 'Ceftriaxone', 'Vancomycin', 'Piperacillin-Tazobactam', 'Ciprofloxacin', 'Cefepime', 'Cotrimoxazole', 'Meropenem']

        out = []
        for idx in range(len(antibiotics)):
            out.append(Resistance(antibioticName= antibiotics[idx], antibioticResistance = output[idx].item()))

        return Resistances(resistances = out)
