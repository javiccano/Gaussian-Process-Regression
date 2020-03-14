# Gaussian Process Regression

## Energy Generation Prediction

The goal of this challenge is to accurately predict the net hourly generated energy of a Combined Cycle Power Plant. In order to do so, five different measured variables that should be related to the output are provided. These variables are for instance the Temperature, Relative humidity among others.

As usual, the provided data has been corrupted with noise, and some values have been lost during their acquisition.

## Solution

A Gaussian Process Regression (GPR) model with a Matérn kernel (nu = 5/2) was implemented to both estimate the output and the missing values of the data.

## Acknowledgements

University Carlos III of Madrid, Data Processing (https://www.kaggle.com/c/uc3m-data-processing/overview/description).
