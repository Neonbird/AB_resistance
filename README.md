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
Modules and libraries for Python (>= 3.6):
* scikit-learn (>= 0.23.1)
* NumPy (>= 1.13.3)
* SciPy (>= 0.19.1)
* joblib (>= 0.11)
* threadpoolctl (>= 2.0.0)



# Database
Initial information about SRA ids of WGS data and information on the presence of antibiotic resistance and the corresponding MICs were obtained from the PATRIC database for *Klebsiella pneumoniae* strains.
WGS data was download from NCBI Sequence Read Archive (SRA).

# Pre-processing of genome sequencing data
* Quality control was carried out using FastQC, results were inspected using MultiQC.
* Adapter sequences adjusted in Trimmomatic;
* Reeds are cleared of point sequencing errors using Bayeshammer __!!! WICH COMMAND !!!__;
* Creating a 10-mer library using Jellyfish __!!! WHICH COMMAND !!!__.

## Embedding of genetic data
As a method of embedding genetic sequences, k-mers counting has shown itself to be the best way. A 10-mer library was created on paired reads using Jellyfish (see comand above). A counting matrix for paired reads was created for gentamicin resistant strains. From this matrix, training and test data sets were generated with dimensions of 1556 × 524 800 and 220 × 524 800, respectively. 

## Create matrix
In order to collect information about 10-mers counts into a single matrix along with the target value - MIC, we used a script  `make_matrix.py ` created in collaboration with a project colleague. This script placed in her repository https://github.com/anastasia2145/Practice-2020/blob/master/make_matrix.py as a part of project from another organisation.


# Regression task: MIC prediction
Due to the large number of features it is impossible to build many models, because not enough memory, so we decided to use several schemes for feature selecting and following regression.

## lasso_regression.py
### Method
One approach was to use logarithmic regression model with L1 regularization (lasso) to extract top of important features, which later planned to be used in random forest. However, we ran out of memory to run a regression on all features. To get around this problem and still use lasso to select features, the following scheme was used:
1) From all features, select the 50,000 most significant using the chi-square criterion on the independence of the samples;
2) Apply a logarithmic regression model with L1 regularization to this 50,000 features to select the most significant features.
For this purposes script `lasso_regression` was written.
### Input format
Input is a two numpy arrays in which row correspond to samples and columns correspond to features, beside of last column that correspond to target (MIC concentration). All values for features should be normalised. One file should be train dataset and another should be test dataset.
#### Example of runnig
``` lasso_regression.py file_train.npy file_test.npy ```

### Output format
A standard output is a text that contains information about statistics such as R-squared (r2) and Root Mean Square Error (RMSE) for lasso on test and training datasets, the number of features selected using lasso, as well as statistics r2 and RMSE for random forest on test and training datasets.

#### Example of output
```
### LOG LASSO REGRESSION ###
Test Lasso r2-score is 0.2
Test Lasso RMSE is 4.5
Train Lasso r2-score is 0.3
Train Lasso RMSE is 3.8
datasets trasformed to 1209 features...

RANDOM FOREST
selected features by lasso: 1209
Test Random forest r2-score is 0.5
Test Random forest RMSE is 2.8
Train Random forest r2-score is 0.9
Train Random forest RMSE is 2.2
DONE
```

## boost.py
### Usage
Does a similar method lasso_regression.py, then train gradient boosting and evaluates MSE.
### Input format
Input two files, a training dataset and a test dataset in the binary form of a numpy module matrix.
#### Example of runnig
```boost.py train.npy test.npy ```
### Output
Returns to MSE console for loss = deviance and for loss = exponential.
```
For example, 
MSE score for XGBoost loss = deviance: 0.5;
MSE score for XGBoost loss = exponential = 0.6.
```

# Classification task: prescence of resistance
## chi2_random_compare.py
### Usage
Evaluates the ROC AUC of a dataset with 10,000 random features (k-mers) and 10,000 attributes selected by the Chi-Square criterion on Random Forest with default parameters from the sklearn module.
### Input format
Input two files, a training dataset and a test dataset in the binary form of a numpy module matrix.
#### Example of runnig
```chi2_random_compare.py -file_name train.npy -file_name_test test.npy ```
### Output
Creates a list written in the binary form of numpy module, the first 10 numbers are responsible for ROC AUC calculated on the features selected by the criterion Chi square, the remaining 10 ROC AUC are responsible for random features.

## chi2_catboost.py
### Usage
The chi-square criterion is applied to each attribute from the dataset and its importance is assessed in relation to the prediction of a target variable, all values are sorted and taken from 1000 to 10000 attributes in steps of 1000. On which the CatBoost algorithm is subsequently applied.
### Input format
Accepts two files, a training dataset and a test dataset in the binary form of a numpy module matrix.
#### Example of runnig
```chi2_catboost.py  -file_name train.npy -file_name_test test.npy ```
### Output
Creates a list recorded in the binary form of the Numpy module with recorded ROC AUC for each number of selected features.

# Additional scripts
## lasso_r2_stats.py
### Usage
This script was used to visualize the change of r2 statistic when adding features in a lasso.
### Input format
Input is a two numpy arrays in which row correspond to samples and columns correspond to features, beside of last column that correspond to target (MIC concentration). All values for features should be normalised. One file should be train dataset and another should be test dataset.
#### Example of runnig
``` lasso_regression.py file_train.npy file_test.npy ```
### Output
A standard output is a text that contains information about statistics such as R-squared (r2) for lasso on test and training datasets and the number of features selected using lasso.
```
### LOG LASSO REGRESSION ###
Test Lasso r2-score is 0.2
Train Lasso r2-score is 0.3
datasets trasformed to 1209 features by lasso ...
DONE
```
In addition, two files are created in the directory from which the script was run:
1) `r2_stats.csv` consisting of two lines. The first line contains information about the value of statistics for the test dataset, and the second for training;
2) `r2_stats.png` is a plot that showing change of statistics r2 (Oy) during selection of features (Ox) by lasso; blue line for train and red line for test.

## prefetch_script.sh
### Usage
This script is written in bash, it downloads the file by SRA identifier, converts it to FASTA format, trimmomatic cuts FASTA file by adapters and at the end.
### Input format
In filename, the program itself needs to include a text file with the SRA list, and then just run it in the console.
#### Example of runnig
```./prefetch_script.sh ```
### Output
Returns the trimmed FASTA files.
