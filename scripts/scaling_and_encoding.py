import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_data(df, columns):
    """
    Normalize specified columns in the dataframe.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        columns (list): List of columns to normalize.
    
    Returns:
        pd.DataFrame: Dataframe with normalized columns.
    """
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def encode_categorical_features(df, columns):
    """
    Encode categorical features using one-hot encoding.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        columns (list): List of categorical columns to encode.
    
    Returns:
        pd.DataFrame: Dataframe with encoded categorical features.
    """
    df = pd.get_dummies(df, columns=columns, drop_first=True)
    return df