# Project goal:
Development of an optimal approach for the analysis of *Klebsiella pneumonia* resistance to known antibiotics by machine learning methods based on genome-wide sequencing data

# Tasks:
1) Search for a suitable database
2) Pre-processing of genome sequencing data
3) Choosing a method for embedding genetic data
4) Development of a model for predicting the presence of antibiotic resistance and a minimum inhibitory concentration based on information on k-measures from WGS readings by machine learning methods.

# System requirements for the developed software
memory
CPU
Python3.6
libaries


# Database
Initial information about SRA ids of WGS data and information on the presence of antibiotic resistance and the corresponding MICs were obtained from the PATRIC database for *Klebsiella pneumoniae* strains.
WGS data was download from NCBI Sequence Read Archive (SRA).

# Pre-processing of genome sequencing data
* Quality control was carried out using FastQC, results were inspected using MultiQC.
* Reeds are cleared of point sequencing errors using Bayeshammer __WICH COMMAND__;
* Creating a 10-mer library using Jellyfish __WHICH COMMAND__.

# Embedding for genetic data
As a method of embedding genetic sequences, k-mers counting has shown itself to be the best way. A 10-mer library was created on paired reads using Jellyfish. A counting matrix for paired reads was created for gentamicin resistant strains. From this matrix, training and test data sets were generated with dimensions of 1556 × 524 800 and 220 × 524 800, respectively. 



