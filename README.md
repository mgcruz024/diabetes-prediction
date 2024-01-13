# diabetes-prediction
ML Prediction for Diabetes data (Kaggle) using Class Imbalance Techniques 


# Diabetes Prediction using Logistic Regression, kNN, Classification Trees and Random Forest

## Overview

This project utilizes the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) from Kaggle, sourced from the Behavioral Risk Factor Surveillance System (BRFSS). BRFSS is an annual health-related telephone survey conducted by the Center of Disease Control and Prevention (CDC).

## Dataset Details

- Original dataset: 441,455 individuals, 330 features.
- Kaggle contributors tailored the dataset to 253,680 survey responses and 21 feature variables.

## Analysis

The dataset includes responses to questions directly posed to participants and calculated variables based on individual responses. With one target class variable and other predictor variables, the project explores prediction models using Logistic Regression, kNN, CART, and Random Forest.

### Risk Factors

The variables considered as risk factors are categorical, mostly falling into binary categories.

### Class Imbalance

The dataset exhibits a target class imbalance with 0's (213,703) and 1's (39,977). To address this, two sampling techniques were applied:

1. **Undersampling:**
   - 0's: 39,977
   - 1's: 39,977

2. **Oversampling using SMOTE:**
   - 0's: 213,703
   - 1's: 199,885

## Data Preprocessing

To ensure model effectiveness, the correlation between independent variables after sampling was verified. The correlation among variables ranged between 0.8 and -0.8.

## Model Building

With a balanced dataset, the project proceeds to build and evaluate predictive models, including Logistic Regression, kNN, CART, and Random Forest.

**Note:** The code and detailed analysis can be found in the Jupyter Notebook File provided in this repository.

## How to Use

1. Clone the repository.
2. Open with Jupyter Notebook to view the code and analysis.
3. Explore the dataset and experiment with the predictive models.

Feel free to contribute or provide feedback!

**Dataset Source:** [Diabetes Health Indicators Dataset on Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
