# CAP_Sleep_Disorders

This repository will include the macrostructural and microstructural (CAP) features extracted from the annotated hypnograms
of the unique sleep disorder database that you find here :
CAP SLEEP DATABASE
https://physionet.org/content/capslpdb/1.0.0/#ref03

We released the original features and the ones weighted with z-score per feature (SLEIS score 1) and with its aggregated version (SLEIS score 2).
We also released the machine learning approach following a nested cross-validation {5,2} on the three experiments:
[1] on the original approach with the healthy controls vs NFLE vs RBD using the original features
[2] employing the individual z-scores (SLEIS score v.1) estimated seaprately for the training-testing set to discriminate NFLE vs RBD
[3] employing the aggregated z-scores (SLEIS score v.2) estimated seaprately for the training-testing set to discriminate NFLE vs RBD

