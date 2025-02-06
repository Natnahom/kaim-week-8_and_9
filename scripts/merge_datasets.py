import pandas as pd

def ip_to_int(ip):
    """
    Convert an IP address to an integer.
    
    Parameters:
        ip (str or float or int): IP address.
    
    Returns:
        int or None: Integer representation of the IP address, or None if invalid.
    """
    # If the input is already an int, return it as is
    if isinstance(ip, int):
        ip = ip

    # Convert float to string and then process
    if isinstance(ip, float):
        ip = str(ip) 
        ip = int(''.join(f'{int(octet):08b}' for octet in ip.split('.')), 2)


    # Quick check to see if it looks like an IP
    # if '.' not in ip:
    #     return None
    
    try:
        return ip
    except (ValueError, TypeError):
        return None
    
def merge_geolocation_data(fraud_df, ip_country_df):
    """
    Merge Fraud_Data.csv with IpAddress_to_Country.csv for geolocation analysis.
    
    Parameters:
        fraud_df (pd.DataFrame): Fraud_Data dataframe.
        ip_country_df (pd.DataFrame): IpAddress_to_Country dataframe.
    
    Returns:
        pd.DataFrame: Merged dataframe.
    """
    # Convert IP addresses to integers
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)
    
    # Drop rows with invalid IP addresses
    fraud_df = fraud_df.dropna(subset=['ip_int'])
    
    # Convert IP bounds to integers
    ip_country_df['lower_bound_int'] = ip_country_df['lower_bound_ip_address'].apply(ip_to_int)
    ip_country_df['upper_bound_int'] = ip_country_df['upper_bound_ip_address'].apply(ip_to_int)
    
    # Drop rows with invalid IP bounds
    ip_country_df = ip_country_df.dropna(subset=['lower_bound_int', 'upper_bound_int'])
    
    # Ensure the columns are of integer type
    fraud_df['ip_int'] = fraud_df['ip_int'].astype(int)
    ip_country_df['lower_bound_int'] = ip_country_df['lower_bound_int'].astype(int)
    ip_country_df['upper_bound_int'] = ip_country_df['upper_bound_int'].astype(int)
    
    # Merge datasets with a left join to preserve NaN in country
    merged_df = pd.merge_asof(
        fraud_df.sort_values('ip_int'),
        ip_country_df.sort_values('lower_bound_int'),
        left_on='ip_int',
        right_on='lower_bound_int',
        direction='backward',
        suffixes=('', '_country')  # Optional: to avoid column name clashes
    )
    
    # If desired, you can fill NaN in the country column with "No Country" or similar
    # merged_df['country'] = merged_df['country'].fillna('No Country')
    
    return merged_df