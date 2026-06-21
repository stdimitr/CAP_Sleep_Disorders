# CAP_Sleep_Disorders

This repository includes the macrostructural and microstructural (CAP) features extracted from the annotated hypnograms
of the unique sleep disorder database that you find here :
CAP SLEEP DATABASE
https://physionet.org/content/capslpdb/1.0.0/#ref03

We released the original features and the ones weighted with z-score per feature (SLEIS score 1) and with its aggregated version (SLEIS score 2).
We also released the machine learning approach following a nested cross-validation {5,2} on the three experiments:

[1] on the original approach with the healthy controls vs NFLE vs RBD using the original features (three classes)

[2] employing the individual z-scores (SLEIS score v.1) estimated seaprately for the training-testing set to discriminate NFLE vs RBD (two classes)

[3] employing the aggregated z-scores (SLEIS score v.2) estimated seaprately for the training-testing set to discriminate NFLE vs RBD (two classes)

The file total_classes.mat records subjects' classes across the cohort of N = 108 subjects. Below, you can find the
correspondence between numbers and sleep disorder groups.

 (1)    n    = 1:16
 
 (2)   brux  = 17:18
 
 (3)   ins   = 19:27  
 
 (4)   narco = 28:32
 
 (5)   nfle  = 33:72
 
 (6)   plm   = 73:82
 
 (7)   rbd   = 83:104
 
 (8)   sbd   = 105:108



Citation :

If you use the meta-data (features) and the code, please cite:
Dimitriadis SI, Salic CI. Cyclic Alternating Patterns (CAP) framework in Sleep Microstructure of Sleep Disorders: Markers of Sleep Instability Using Healthy Controls as Reference.
doi: https://doi.org/10.1101/2025.10.13.25337880
URL : https://www.medrxiv.org/content/10.1101/2025.10.13.25337880v2 


 
 
