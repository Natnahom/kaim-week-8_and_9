import pandas as pd

def ip_to_int(ip):
    """
    Convert an IP address to an integer.
    
    Parameters:
        ip (str): IP address.
    
    Returns:
        int: Integer representation of the IP address.
    """
    return int(''.join(f'{int(octet):08b}' for octet in ip.split('.')), 2)

def merge_geolocation_data(data1, data2):
    """
    Merge Fraud_Data.csv with IpAddress_to_Country.csv for geolocation analysis.
    
    Parameters:
        fraud_df (pd.DataFrame): Fraud_Data dataframe.
        ip_country_df (pd.DataFrame): IpAddress_to_Country dataframe.
    
    Returns:
        pd.DataFrame: Merged dataframe.
    """
    # Convert IP addresses to integers
    data1['ip_int'] = data1['ip_address'].apply(ip_to_int)
    data2['lower_bound_int'] = data2['lower_bound_ip_address'].apply(ip_to_int)
    data2['upper_bound_int'] = data2['upper_bound_ip_address'].apply(ip_to_int)
    
    # Merge datasets
    merged_df = pd.merge_asof(
        data1.sort_values('ip_int'),
        data2.sort_values('lower_bound_int'),
        left_on='ip_int',
        right_on='lower_bound_int',
        direction='backward'
    )
    
    return merged_df