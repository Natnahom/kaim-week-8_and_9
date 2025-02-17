## This is week-8 and 9 of 10 academy

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

# Task 3: Model Evaluation and Interpretation

## Overview

This project focuses on evaluating and interpreting machine learning models for fraud detection using SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations). The goal is to understand the model's decisions and enhance trust in the predictions made.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Interpretation of Results](#interpretation-of-results)
- [Dependencies](#dependencies)
- [License](#license)

## Installation

To run this project, you'll need to set up your environment with the required libraries. You can do this using pip. Create a virtual environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage
1. Load the Dataset: Ensure your dataset is prepared and loaded correctly into the script.
2. Train Models: The script includes functions to train various machine learning models on the dataset.
3. Evaluate Models: Run the evaluation function to assess model performance metrics such as accuracy, precision, and recall.
4. Generate SHAP Values: Use the shap_expl function to generate SHAP values and visualize the feature importances.
5. Create LIME Explanations: Use the lime_expl function to generate local explanations for specific predictions.

## Model Evaluation
The evaluation function calculates various metrics, including:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
These metrics help assess the performance of the models in detecting fraudulent transactions.

## Interpretation of Results
SHAP and LIME provide insights into model behavior:

SHAP: Offers global insights into feature importance across all predictions and local insights for individual predictions.
LIME: Focuses on explaining individual predictions, highlighting which features contributed most to the model's decision.
Dependencies
- pandas
- numpy
- scikit-learn
- shap
- lime
- matplotlib
- seaborn

Make sure to install these libraries using the requirements file.
Then follow the modelTraining.ipynb

# Task 4: Fraud Detection API

## Overview

Task 4 involves developing an API for fraud detection in financial transactions. The primary goal is to create a robust, scalable web service that can analyze transaction data in real-time and predict whether a transaction is fraudulent. The API utilizes machine learning models trained on historical transaction data to make these predictions.

## Objectives

- **Implement a RESTful API**: Create endpoints that allow users to submit transaction data and receive fraud predictions.
- **Machine Learning Integration**: Train and integrate a machine learning model capable of identifying fraudulent transactions based on features such as transaction amount, user behavior, and timestamps.
- **Dockerize the Application**: Use Docker to containerize the application, making it easy to deploy in different environments.
- **Logging and Monitoring**: Implement logging to track requests and responses for auditing and monitoring purposes.

## Features

- **Input Validation**: Ensure that incoming transaction data is validated before processing.
- **Real-time Predictions**: Provide immediate feedback on whether a transaction is likely to be fraudulent.
- **Scalability**: The application is designed to handle multiple requests simultaneously, making it suitable for high-traffic environments.
- **Comprehensive Logging**: Track API usage and errors through a detailed logging system.

## Getting Started

### Prerequisites

- Docker
- Python 3.9+
- Flask
- Required Python libraries (listed in `requirements.txt`)

### To run the api
- Use postman if needed or

- cd app
- flask --app serve_model app

### Installation

1. **Clone the repository**:

2. **Build the Docker image**:
- docker build -t fraud-detection-api .

3. **Run the Docker container**:
- docker run -p 5000:5000 fraud-detection-api

### API Endpoints
1. POST /predict
- Description: Analyzes a transaction and predicts if it is fraudulent.
- Request Body:
{
  "transaction_id": "123456",
  "amount": 100.0,
  "timestamp": "2023-01-01T12:00:00Z",
  "user_id": "user123"
}
- Response:
{
  "transaction_id": "123456",
  "is_fraud": true,
  "confidence_score": 0.95
}

## Logging
The API includes logging functionality to monitor requests and responses. Logs are saved to the app/logs directory, which can be useful for debugging and analyzing the performance of the model.

# Task 5 - Fraud Detection Dashboard with Flask and Dash
This project involves creating an interactive dashboard using Flask and Dash to visualize fraud insights from a dataset. The Flask backend serves data from a CSV file, while Dash handles the frontend visualizations.

## Project Structure
kaim-week-8_and_9/
  dashboard/
  │
  ├── app.py                  # Flask backend
  ├── dashboard.py             # Dash frontend
  ├── templates/
  │   └── index.html          # HTML template for the dashboard
  ├── static/
  │   └── style.css          # Custom CSS for styling the dashboard
  requirements.txt        # Python dependencies

## Features
1. Flask Backend: Serves data via API endpoints.

- /api/summary: Returns summary statistics (total transactions, fraud cases, fraud percentage).

- /api/trends: Returns fraud trends over time.

- /api/product_value: Returns product value distribution of fraud cases.

- /api/age: Returns age distribution of fraud cases.

- /api/geographic: Returns geographic distribution of fraud cases.

- /api/device_browser: Returns fraud cases by device and browser.

2. Dash Frontend: Visualizes data fetched from the Flask backend.

3. Summary Boxes: Display total transactions, fraud cases, and fraud percentage.

4. Line Chart: Shows the number of detected fraud cases over time.

5. Bar Chart: Compares the number of fraud cases across different devices and browsers.

6. Geographic Map: Displays fraud cases by location.

## Installation
1. Clone the Repository:

2. Install Dependencies:

- pip install -r requirements.txt
Run the Flask Backend:

- python app.py
Run the Dash Frontend:

- python dashboard.py
Access the Dashboard:

Flask app: http://127.0.0.1:5000/

Dash app: http://127.0.0.1:8050/

## Usage
Flask Backend: The backend serves data through API endpoints. You can test these endpoints by visiting them in your browser or using tools like curl or Postman.

Dash Frontend: The frontend fetches data from the Flask backend and displays it in an interactive dashboard. You can interact with the charts and visualizations directly in the browser.

## Customization
Data: Replace Data/Fraud_Data.csv with your own dataset. Ensure the column names match those expected by the Flask backend.

Styling: Modify static/style.css to customize the appearance of the dashboard.

Additional Features: Extend the Flask backend or Dash frontend to include more visualizations or functionality as needed.

## Troubleshooting
Data Not Displaying: Ensure that the Flask backend is running and accessible. Check the browser console for any errors related to fetching data.

Styling Issues: Verify that the path to styles.css is correct and the file is accessible.

Endpoint Errors: Ensure that the endpoint names in the Flask backend match those used in the Dash frontend.

Author: Natnahom Asfaw
Date: 06/02/2025