import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def handle_missing_val(df, strategy='drop', fill_value=None):
    """
    Handle missing values in the dataset.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        strategy (str): Strategy to handle missing values ('drop' or 'impute').
        fill_value: Value to use for imputation (if strategy is 'impute').
    
    Returns:
        pd.DataFrame: Dataframe with missing values handled.
    """
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'impute':
        df = df.fillna(fill_value)
    
    return df
    

def data_cleaning(df):
    """
    Clean the dataset by removing duplicates and correcting data types.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
    
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = df.drop_duplicates()

    if 'purchase_time' in df.columns:
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    if 'signup_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'])
    
    return df

def univariate(df, column, size1=10, size2=6):
    """
    Perform univariate analysis on a specific column.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        column (str): Column to analyze.
    """
    plt.figure(figsize=(size1, size2))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()

def bivariate(df, col1, col2, size1=10, size2=6):
    """
    Perform bivariate analysis between two columns.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        col1 (str): First column.
        col2 (str): Second column.
    """
    plt.figure(figsize=(size1, size2))
    sns.scatterplot(data=df, x=col1, y=col2)
    plt.title(f"{col1} vs {col2}")
    plt.show()