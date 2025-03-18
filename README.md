# Multi-label classification model with masked loss for prediction of antimicrobial resistance using MALDI-TOF MS data

Analysis code and results for antimicrobial resistance prediction model for clinical MALDI-TOF MS data.
Furthermore, this project cover in-silico analysis for antimicrobial resistance prediction from mixed MALDI-TOF MS data profiles.

This repository is a work in progress.

## Analysis codes (Jupyter notebook examples)

| Title & Link | Information |
| ------------- | ------------- |
| [Data preprocessing](https://github.com/iron-lion/maldi2resistance/blob/main/notebooks/0_msdata_preprocessing.ipynb)  | Code for data preprocessing (binning) |
| [Assymetric loss w. ResMLP](https://github.com/iron-lion/maldi2resistance/blob/main/notebooks/multilabel_classification_assymetric-loss/1_assymetric_loss_test_ResMLP-multilabel.ipynb) | Model evaluation (5-fold cv), [Results](https://github.com/iron-lion/maldi2resistance/tree/main/notebooks/multilabel_classification_assymetric-loss/results_ResMLP-multilabel)|
| [Assymetric loss w. MLP](https://github.com/iron-lion/maldi2resistance/blob/main/notebooks/multilabel_classification_assymetric-loss/2_assymetric_loss_test_MLP.ipynb) | Model evaluation (5-fold cv), [Results](https://github.com/iron-lion/maldi2resistance/tree/main/notebooks/multilabel_classification_assymetric-loss/results_MLP) |
| [Assymetric loss w. multimodalAMR](https://github.com/iron-lion/maldi2resistance/blob/main/notebooks/multilabel_classification_assymetric-loss/3_assymetric_loss_on_multimodalAMR.ipynb) | Model evaluation (5-fold cv), [Results](https://github.com/iron-lion/maldi2resistance/tree/main/notebooks/multilabel_classification_assymetric-loss/multimodalAMR_5cv-UMG/csv) |
| [MultimodalAMR run](https://github.com/iron-lion/maldi2resistance/blob/main/notebooks/comparison/1_multimodalAMR.ipynb) | Model evaluation (5-fold cv), [Results](https://github.com/iron-lion/maldi2resistance/tree/main/notebooks/comparison/) |
| [DualBranch run](https://github.com/iron-lion/maldi2resistance/blob/main/notebooks/comparison/2_dual_branch.ipynb) | Model evaluation (5-fold cv), [Results](https://github.com/iron-lion/maldi2resistance/tree/main/notebooks/comparison/) |

## Further reading/projects
 - maldi-learn: https://github.com/BorgwardtLab/maldi-learn
 - maldi_amr: https://github.com/BorgwardtLab/maldi_amr
 - DRIAMS dataset: https://datadryad.org/dataset/doi:10.5061/dryad.bzkh1899q
 - MS-UMG dataset: https://doi.org/10.5281/zenodo.13911744
 - Data Heterogeneity in Clinical MALDI-TOF Mass Spectra Profiles : https://www.biorxiv.org/content/10.1101/2024.10.18.617592v1.abstract
 - Multimodal AMR work: https://github.com/BorgwardtLab/MultimodalAMR/blob/main/multimodal_amr/models/classifier.py
 - This repository is forked and modified based on the maldi2resistance project, https://github.com/JanNiklasWeder/maldi2resistance.
