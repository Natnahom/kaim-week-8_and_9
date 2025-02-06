def create_features(df):
    """
    Create new features for fraud detection.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
    
    Returns:
        pd.DataFrame: Dataframe with new features.
    """
    # Time-based features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # Transaction frequency (number of transactions per user)
    df['transaction_frequency'] = df.groupby('user_id')['user_id'].transform('count')
    
    # Transaction velocity (time between transactions)
    df['time_since_last_transaction'] = df.groupby('user_id')['purchase_time'].diff().dt.total_seconds()
    
    return df