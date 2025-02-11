## This is week-7 and 8 of 10 academy

# Task 1: Data Analysis and Preprocessing

Task 1 consists of the fraud detection project at Adey Innovations Inc. The goal of this task is to analyze and preprocess transaction data to prepare it for building machine learning models for fraud detection.

## Overview
The task involves the following steps:

1. Handling Missing Values: Impute or drop missing values in the dataset.

2. Data Cleaning: Remove duplicates and correct data types.

3. Exploratory Data Analysis (EDA): Perform univariate and bivariate analysis to understand the data.

4. Merge Datasets for Geolocation Analysis: Merge Fraud_Data.csv with IpAddress_to_Country.csv to map IP addresses to countries.

5. Feature Engineering: Create new features such as transaction frequency, velocity, and time-based features.

6. Normalization and Scaling: Normalize numerical features for better model performance.

7. Encode Categorical Features: Convert categorical variables into numerical representations using one-hot encoding.

## Datasets
The following datasets are used in this task:

- Fraud_Data.csv: Contains e-commerce transaction data with features such as user_id, purchase_time, purchase_value, ip_address, and class (fraud indicator).

- IpAddress_to_Country.csv: Maps IP address ranges to countries.

- creditcard.csv: Contains anonymized bank transaction data for fraud detection.

## Code Structure
The code is modular and organized into functions for each preprocessing step. Below is the structure of the code:

1. Handle Missing Values
Function: handle_missing_values

Description: Imputes or drops missing values in the dataset.

2. Data Cleaning
Function: clean_data

Description: Removes duplicates and corrects data types (e.g., converting timestamps to datetime).

3. Exploratory Data Analysis (EDA)
Functions:

- univariate_analysis: Analyzes the distribution of a single column.

- bivariate_analysis: Analyzes the relationship between two columns.

4. Merge Datasets for Geolocation Analysis
Functions:

- ip_to_int: Converts IP addresses to integers.

merge_geolocation_data: Merges Fraud_Data.csv with IpAddress_to_Country.csv based on IP addresses.

5. Feature Engineering
Function: create_features

Description: Creates new features such as hour_of_day, day_of_week, transaction_frequency, and time_since_last_transaction.

6. Normalization and Scaling
Function: normalize_data

Description: Normalizes numerical features using StandardScaler.

7. Encode Categorical Features
Function: encode_categorical_features

Description: Encodes categorical variables using one-hot encoding.

## How to Run the Code
1. Clone the repo
2. Install the required dependencies:

    - pip install -r requirements.txt
3. Follow analysis1.ipynb file to run each functions

# Task 2: Model Implementation and Evaluation

In Task 2, we focus on implementing and evaluating various machine learning models to detect fraudulent transactions. The steps involved include:

1. **Data Preparation**: 
   - Load the dataset and preprocess it, including handling missing values, encoding categorical features, and normalizing numerical features.
   - Split the dataset into training and testing sets to evaluate model performance on unseen data.

2. **Model Training**:
   - Define a set of models, including both traditional machine learning algorithms and deep learning models.

   - For each model, fit the training data and adjust hyperparameters as necessary. This includes:

     - Logistic Regression: A linear model that predicts the probability of fraud based on input features.

     - Decision Trees: A non-linear model that builds a tree-like structure for making decisions based on feature splits.

     - Random Forest: An ensemble method that combines multiple decision trees for improved accuracy and robustness.

     - Gradient Boosting: Another ensemble method that builds models sequentially, each trying to correct the errors of the previous ones.

     - Multi-layer Perceptron (MLP): A neural network model with one or more hidden layers to capture complex relationships.

     - Convolutional Neural Network (CNN): A deep learning model often used for image processing but adapted here for feature extraction.
     - Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) networks: Models designed to handle sequential data, capturing temporal dependencies.

3. **Model Evaluation**:
   - Evaluate each model using metrics such as accuracy, precision, recall, and F1-score.
   - Generate confusion matrices to visualize the performance of each model in predicting fraud versus non-fraud cases.
   - Log the results using MLflow for easy comparison and tracking of model performance across different runs.

Through this task, we aim to identify the most effective model for detecting fraud, balancing accuracy with the ability to generalize to new, unseen data.

This project implements various machine learning models for fraud detection, including traditional models and neural networks. The goal is to evaluate and compare these models based on their performance metrics.

## Features

- Implementation of the following models:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Multi-layer Perceptron (MLP)
  - Convolutional Neural Network (CNN)
  - Recurrent Neural Network (RNN)
  - Long Short-Term Memory (LSTM)

- Model evaluation using metrics such as accuracy, confusion matrix, and classification report.
- Logging of model results using MLflow for easier tracking of experiments.

## Installation

1. Clone the repository:

2. Create a virtual environment (optional but recommended):
- python -m venv venv
source venv/bin/activate  `venv\Scripts\activate`

Install the required packages:
- pip install -r requirements.txt

## Usage
- Follow the modelTraining.ipynb

Author: Natnahom Asfaw
Date: 06/02/2025