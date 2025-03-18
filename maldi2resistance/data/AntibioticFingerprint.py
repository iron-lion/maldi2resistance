from functools import lru_cache
from typing import Union

import numpy as np
from PIL.PngImagePlugin import PngImageFile
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdFingerprintGenerator
from rdkit.sping import PIL


class FingerprintLookup:
    drug_name2Smiles = {
        "Amikacin": "C1C(C(C(C(C1NC(=O)C(CCN)O)OC2C(C(C(C(O2)CO)O)N)O)O)OC3C(C(C(C(O3)CN)O)O)O)N",
        "Amoxicillin-Clavulanic acid": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C.C1C2N(C1=O)C(C(=CCO)O2)C(=O)O",
        "Ampicillin": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=CC=C3)N)C(=O)O)C",
        # This is the result of combining both individual SMILES with a dot
        "Ampicillin-Amoxicillin": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=CC=C3)N)C(=O)O)C.CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C",

        "Aztreonam": "CC1C(C(=O)N1S(=O)(=O)O)NC(=O)C(=NOC(C)(C)C(=O)O)C2=CSC(=N2)N",
        "Benzylpenicillin": "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C",
        "Cefazolin": "CC1=NN=C(S1)SCC2=C(N3C(C(C3=O)NC(=O)CN4C=NN=N4)SC2)C(=O)O",
        "Cefepime": "C[N+]1(CCCC1)CC2=C(N3C(C(C3=O)NC(=O)C(=NOC)C4=CSC(=N4)N)SC2)C(=O)[O-]",
        "Cefpodoxime": "COCC1=C(N2C(C(C2=O)NC(=O)C(=NOC)C3=CSC(=N3)N)SC1)C(=O)O",
        "Ceftazidime": "CC(C)(C(=O)O)ON=C(C1=CSC(=N1)N)C(=O)NC2C3N(C2=O)C(=C(CS3)C[N+]4=CC=CC=C4)C(=O)[O-]",
        "Ceftriaxone": "CN1C(=NC(=O)C(=O)N1)SCC2=C(N3C(C(C3=O)NC(=O)C(=NOC)C4=CSC(=N4)N)SC2)C(=O)O",
        "Cefuroxime": "CON=C(C1=CC=CO1)C(=O)NC2C3N(C2=O)C(=C(CS3)COC(=O)N)C(=O)O",
        "Ciprofloxacin":"C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",
        "Clarithromycin": "CCC1C(C(C(C(=O)C(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)OC)C)C)O)(C)O",
        "Clindamycin": "CCCC1CC(N(C1)C)C(=O)NC(C2C(C(C(C(O2)SC)O)O)O)C(C)Cl",
        "Colistin": "CCC(C)CCCC(=O)NC(CCN)C(=O)NC(C(C)O)C(=O)NC(CCN)C(=O)NC1CCNC(=O)C(NC(=O)C(NC(=O)C(NC(=O)C(NC(=O)C(NC(=O)C(NC1=O)CCN)CC(C)C)CC(C)C)CCN)CCN)C(C)O",
        "Cotrimoxazole": "CC1=CC(=NO1)NS(=O)(=O)C2=CC=C(C=C2)N.COC1=CC(=CC(=C1OC)OC)CC2=CN=C(N=C2N)N",
        "Ertapenem": "CC1C2C(C(=O)N2C(=C1SC3CC(NC3)C(=O)NC4=CC=CC(=C4)C(=O)O)C(=O)O)C(C)O",
        "Erythromycin": "CCC1C(C(C(C(=O)C(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)O)(C)O",
        "Fosfomycin": "CC1C(O1)P(=O)(O)O",
        "Fosfomycin-Trometamol": "CC1C(O1)P(=O)(O)O.C(C(CO)(CO)N)O",
        "Fusidic acid": "CC1C2CCC3(C(C2(CCC1O)C)C(CC4C3(CC(C4=C(CCC=C(C)C)C(=O)O)OC(=O)C)C)O)C",
        "Gentamicin": "CC(C1CCC(C(O1)OC2C(CC(C(C2O)OC3C(C(C(CO3)(C)O)NC)O)N)N)N)NC",
        "Imipenem": "CC(C1C2CC(=C(N2C1=O)C(=O)O)SCCN=CN)O",
        "Levofloxacin": "CC1COC2=C3N1C=C(C(=O)C3=CC(=C2N4CCN(CC4)C)F)C(=O)O",
        "Meropenem": "CC1C2C(C(=O)N2C(=C1SC3CC(NC3)C(=O)N(C)C)C(=O)O)C(C)O",
        "Mupirocin": "CC(C1C(O1)CC2COC(C(C2O)O)CC(=CC(=O)OCCCCCCCCC(=O)O)C)C(C)O",
        "Nitrofurantoin": "C1C(=O)NC(=O)N1N=CC2=CC=C(O2)[N+](=O)[O-]",
        "Norfloxacin": "CCN1C=C(C(=O)C2=CC(=C(C=C21)N3CCNCC3)F)C(=O)O",
        "Oxacillin": "CC1=C(C(=NO1)C2=CC=CC=C2)C(=O)NC3C4N(C3=O)C(C(S4)(C)C)C(=O)O",


        # assumed to be pennicillin-G therefore identical to Benzylpenicillin
        "Penicillin": "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C",


        "Piperacillin-Tazobactam": "CCN1CCN(C(=O)C1=O)C(=O)NC(C2=CC=CC=C2)C(=O)NC3C4N(C3=O)C(C(S4)(C)C)C(=O)[O-].CC1(C(N2C(S1(=O)=O)CC2=O)C(=O)O)CN3C=CN=N3",


        # pubchem diescribes it as mixture of Polymyxin B1 and Polymyxin B2
        # SMILES is therefore a combination of those
        # pubchem des not provide those SMILES they are taken from ChEBI
        "Polymyxin B": "CC[C@@H](C)CCCCC(=O)N[C@@H](CCN)C(=O)N[C@@H]([C@@H](C)O)C(=O)N[C@@H](CCN)C(=O)N[C@H]1CCNC(=O)[C@@H](NC(=O)[C@H](CCN)NC(=O)[C@H](CCN)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](Cc2ccccc2)NC(=O)[C@H](CCN)NC1=O)[C@@H](C)O.CC(C)CCCCC(=O)N[C@@H](CCN)C(=O)N[C@@H]([C@@H](C)O)C(=O)N[C@@H](CCN)C(=O)N[C@H]1CCNC(=O)[C@@H](NC(=O)[C@H](CCN)NC(=O)[C@H](CCN)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](Cc2ccccc2)NC(=O)[C@H](CCN)NC1=O)[C@@H](C)O",


        "Rifampicin": "CC1C=CC=C(C(=O)NC2=C(C(=C3C(=C2O)C(=C(C4=C3C(=O)C(O4)(OC=CC(C(C(C(C(C(C1O)C)O)C)OC(=O)C)C)OC)C)C)O)O)C=NN5CCN(CC5)C)C",
        "Teicoplanin": "CCCCCCCCCC(=O)NC1C(C(C(OC1OC2=C3C=C4C=C2OC5=C(C=C(C=C5)C(C6C(=O)NC(C7=C(C(=CC(=C7)O)OC8C(C(C(C(O8)CO)O)O)O)C9=C(C=CC(=C9)C(C(=O)N6)NC(=O)C4NC(=O)C1C2=CC(=CC(=C2)OC2=C(C=CC(=C2)C(C(=O)NC(CC2=CC(=C(O3)C=C2)Cl)C(=O)N1)N)O)O)O)C(=O)O)OC1C(C(C(C(O1)CO)O)O)NC(=O)C)Cl)CO)O)O",
        "Tetracycline": "CC1(C2CC3C(C(=O)C(=C(C3(C(=O)C2=C(C4=C1C=CC=C4O)O)O)O)C(=O)N)N(C)C)O",
        "Tobramycin": "C1C(C(C(C(C1N)OC2C(C(C(C(O2)CO)O)N)O)O)OC3C(CC(C(O3)CN)O)N)N",
        "Vancomycin": "CC1C(C(CC(O1)OC2C(C(C(OC2OC3=C4C=C5C=C3OC6=C(C=C(C=C6)C(C(C(=O)NC(C(=O)NC5C(=O)NC7C8=CC(=C(C=C8)O)C9=C(C=C(C=C9O)O)C(NC(=O)C(C(C1=CC(=C(O4)C=C1)Cl)O)NC7=O)C(=O)O)CC(=O)N)NC(=O)C(CC(C)C)NC)O)Cl)CO)O)O)(C)N)O"
    }

    def __init__(self, fprint_bits:int = 1024):

        self.fpgen = rdFingerprintGenerator.GetMorganGenerator(fpSize=fprint_bits)
        pass

    def get_smiles(self, antibiotic: Union[str,list]) -> str:
        if isinstance(antibiotic, list):
            smiles = []
            for i in antibiotic:
                smiles.append(self.drug_name2Smiles[i])
            return smiles

        return self.drug_name2Smiles[antibiotic]

    @lru_cache(maxsize=64)
    def get_MorganFingerprint(self, antibiotic: str) -> np.ndarray:
        smiles = self.drug_name2Smiles[antibiotic]
        m = Chem.MolFromSmiles(smiles)

        fprint = AllChem.GetMorganFingerprintAsBitVect(m, radius=3, nBits=1024)
        vec = np.array(fprint)

        return vec

    @lru_cache(maxsize=64)
    def antibiotic_image(self,antibiotic:str) -> PngImageFile:
        smiles = self.drug_name2Smiles[antibiotic]
        m = Chem.MolFromSmiles(smiles)

        img = Draw.MolToImage(m)
        return img
