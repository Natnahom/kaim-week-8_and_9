from sklearn.model_selection import train_test_split
from scripts.cleaning_and_EDA import data_cleaning
from scripts.scaling_and_encoding import encode_categorical_features
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def data_preparation(df):
    """
    Feature and Target Separation.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
    
    Returns:
        train_test_split
    """
    # Convert timestamp columns to datetime
    df = data_cleaning(df)

    # Extract features from datetime
    if 'signup_time' in df.columns:
        df['signup_year'] = df['signup_time'].dt.year
        df['signup_month'] = df['signup_time'].dt.month
        df['signup_day'] = df['signup_time'].dt.day
        df['signup_hour'] = df['signup_time'].dt.hour

    if 'purchase_time' in df.columns:
        df['purchase_year'] = df['purchase_time'].dt.year
        df['purchase_month'] = df['purchase_time'].dt.month
        df['purchase_day'] = df['purchase_time'].dt.day
        df['purchase_hour'] = df['purchase_time'].dt.hour

    # Drop original timestamp columns
    df = df.drop(columns=['signup_time', 'purchase_time'], errors='ignore')

    encoded_columns = ['device_id', 'source', 'browser', 'sex']

    # Convert categorical columns to numeric
    for col in encoded_columns:
        if col in df.columns:
            # For other columns, apply one-hot encoding
            df = pd.get_dummies(df, columns=[col], drop_first=True, sparse=True)

    X = df.drop(columns=['class']) if 'class' in df.columns else df.drop(columns=['Class'])
    y = df['class'] if 'class' in df.columns else df['Class']

    return train_test_split(X, y, test_size=0.2, random_state=42)

