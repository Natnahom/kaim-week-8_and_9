import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten

def define_models():
    """
    Define and return a dictionary of machine learning models.
    The models include both traditional algorithms and placeholders for deep learning models.
    """
    return {
        'Logistic Regression': LogisticRegression(max_iter=100),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'MLP': MLPClassifier(max_iter=100),
        # 'CNN': None,  # Placeholder for CNN
        # 'RNN': None,  # Placeholder for RNN
        'LSTM': None  # Placeholder for LSTM
    }

def train_cnn(X_train, y_train, X_test, y_test):
    """
    Train a Convolutional Neural Network (CNN) on the training data.
    
    Parameters:
    - X_train: Training features (Pandas DataFrame or NumPy array)
    - y_train: Training labels (Pandas Series or NumPy array)
    - X_test: Test features (Pandas DataFrame or NumPy array)
    - y_test: Test labels (Pandas Series or NumPy array)
    
    Returns:
    - y_pred: Predictions for the test set
    - model: The trained CNN model
    """
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model with Adam optimizer and binary cross-entropy loss
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Convert DataFrames to NumPy arrays and reshape input data for the CNN
    X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Train the model
    model.fit(X_train_reshaped, y_train, epochs=10, batch_size=64, verbose=0)
    
    # Make predictions on the test set
    y_pred = (model.predict(X_test_reshaped) > 0.5).astype("int32")
    
    return y_pred, model

def train_rnn(X_train, y_train, X_test, y_test):
    """
    Train a Recurrent Neural Network (RNN) on the training data.
    
    Parameters:
    - X_train: Training features (Pandas DataFrame or NumPy array)
    - y_train: Training labels (Pandas Series or NumPy array)
    - X_test: Test features (Pandas DataFrame or NumPy array)
    - y_test: Test labels (Pandas Series or NumPy array)
    
    Returns:
    - y_pred: Predictions for the test set
    - model: The trained RNN model
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.5))
    model.add(LSTM(50))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Convert DataFrames to NumPy arrays and reshape input data for the RNN
    X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Train the model
    model.fit(X_train_reshaped, y_train, epochs=10, batch_size=64, verbose=0)
    
    # Make predictions on the test set
    y_pred = (model.predict(X_test_reshaped) > 0.5).astype("int32")
    
    return y_pred, model

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate the performance of different models on the test set.
    
    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_test: Test features
    - y_test: Test labels
    - models: A dictionary of models to evaluate
    
    Returns:
    - results: A dictionary containing performance metrics for each model
    """
    results = {}
    for name, model in models.items():
        if model is None:  # Skip placeholders
            if name == 'CNN':
                y_pred, model = train_cnn(X_train, y_train, X_test, y_test)
            elif name == 'RNN':
                y_pred, model = train_rnn(X_train, y_train, X_test, y_test)
        else:
            model.fit(X_train, y_train)  # Fit the traditional model
            y_pred = model.predict(X_test)  # Make predictions

        # Flatten y_pred if necessary
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()

        # Store evaluation metrics
        results[name] = {
            'report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'accuracy': np.mean(y_pred == y_test)
        }
    return results

def log_results(models, X_train, y_train, X_test, y_test):
    """
    Log model performance metrics using MLflow.
    
    Parameters:
    - models: A dictionary of models
    - X_train: Training features
    - y_train: Training labels
    - X_test: Test features
    - y_test: Test labels
    """
    with mlflow.start_run():
        for name, model in models.items():
            if model is None:  # Handle CNN and RNN separately
                if name == 'CNN':
                    y_pred, model = train_cnn(X_train, y_train, X_test, y_test)
                elif name == 'RNN':
                    y_pred, model = train_rnn(X_train, y_train, X_test, y_test)
            else:
                model.fit(X_train, y_train)  # Fit the traditional model
                y_pred = model.predict(X_test)

            # Log the model and its performance
            mlflow.sklearn.log_model(model, name)
            accuracy = np.mean(y_pred == y_test)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_param("model_name", name)
