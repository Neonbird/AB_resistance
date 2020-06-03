# Project goal:
Development of an optimal approach for the analysis of *Klebsiella pneumonia* resistance to known antibiotics by machine learning methods based on genome-wide sequencing data

# Tasks:
1) Search for a suitable database
2) Pre-processing of genome sequencing data
3) Choosing a method for embedding genetic data
4) Development of a model for predicting the presence of antibiotic resistance and a minimum inhibitory concentration based on information on k-mers rom WGS data using machine learning methods.

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
* Adapter sequences adjusted in Trimmomatic;
* Reeds are cleared of point sequencing errors using Bayeshammer __WICH COMMAND__;
* Creating a 10-mer library using Jellyfish __WHICH COMMAND__.

## Embedding of genetic data
As a method of embedding genetic sequences, k-mers counting has shown itself to be the best way. A 10-mer library was created on paired reads using Jellyfish (see comand above). A counting matrix for paired reads was created for gentamicin resistant strains. From this matrix, training and test data sets were generated with dimensions of 1556 × 524 800 and 220 × 524 800, respectively. 

# Regression task: MIC prediction
Due to the large number of features it is impossible to build many models, because not enough memory, so we decided to use the following scheme for Feature selecting:
1) From all features, select the 150,000 most significant using the chi-square criterion on the independence of the samples;
2) Apply a logarithmic regression model with L1 regularization to this 150,000 features to select the most significant features.
For this purposes 
## lasso_regression.py
### 


