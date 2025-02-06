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

Author: Natnahom Asfaw
Date: 06/02/2025