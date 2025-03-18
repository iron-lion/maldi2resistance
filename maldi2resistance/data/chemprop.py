from typing import List

from chemprop.data import collate_batch, MoleculeDataset, MoleculeDatapoint
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from torch.utils.data import default_collate

from maldi2resistance.data.AntibioticFingerprint import FingerprintLookup
from maldi2resistance.data.driams import Driams

featurizer = SimpleMoleculeMolGraphFeaturizer()


def collate(input:List):
    """
    Naive collate function for use with ChemProp; a more efficient version can be found in the CachedChempropCollate class.
    """

    x, label, drug = zip(*input)
    mol_set = MoleculeDataset(drug, featurizer)

    return default_collate(list(zip(x, label))), collate_batch(mol_set)


class CachedChempropCollate():
    """
    A cached collate function that translates a list of positions corresponding to the position
    of the antibiotic in the 'driams.selected_antibiotics' list into a chemprop TrainingBatch.
    This involves pre-calculating the Chemprop dates of the antibiotics during initialization
    and then only looking them up.

    Note:
        Do not use prepare4chemprop for DriamsSingleAntibiotic when using this class! The collate function needs an
        integer representing the antibiotic and will access based on that integer the precalculate Chemprop datums.
    """
    def __init__(self, driams:Driams):

        fingerprint_lookup = FingerprintLookup()
        drugs_smiles = fingerprint_lookup.get_smiles(driams.selected_antibiotics)
        mols = [MoleculeDatapoint.from_smi(smiles) for smiles in drugs_smiles]
        self.mol_set = MoleculeDataset(mols, featurizer)
        self.mol_set.cache = True

    def collate(self, input: List):
        """
        Collate function for use with a pytorch dataloader. When iterating over this,
        the returned values can be directly accepted as follows:

        Example
        -------

        >>> ((spectrum, label), drug) in data_loader:
        >>>     continue

        """

        x, label, drug_poses = zip(*input)
        mols = [self.mol_set[drug_pos] for drug_pos in drug_poses]

        return default_collate(list(zip(x, label))), collate_batch(mols)