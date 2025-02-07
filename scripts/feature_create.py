import pandas as pd

def create_features(df):
    """
    Create new features for fraud detection.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
    
    Returns:
        pd.DataFrame: Dataframe with new features.
    """
    # Ensure purchase_time is a datetime object
    df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')

    # Time-based features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # Transaction frequency (number of transactions per user)
    df['transaction_frequency'] = df.groupby('user_id')['user_id'].transform('count')
    
    # Transaction velocity (time between transactions) using shift
    previous_time = df.groupby('user_id')['purchase_time'].shift(1)
    df['time_since_last_transaction'] = (df['purchase_time'] - previous_time).dt.total_seconds()

    # Fill NaN values for the first transaction
    df['time_since_last_transaction'] = df['time_since_last_transaction'].fillna(0)

    return df