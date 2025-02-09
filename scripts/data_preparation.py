from sklearn.model_selection import train_test_split


def data_preparation(df):
    """
    Feature and Target Separation.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
    
    Returns:
        train_test_split
    """
    X = df.drop(columns=['class']) if 'class' in df.columns else df.drop(columns=['Class'])
    y = df['class'] if 'class' in df.columns else df['Class']

    return train_test_split(X, y, test_size=0.2, random_state=42)
