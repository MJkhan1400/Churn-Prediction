import pandas as pd
import numpy as np

def clean_data(data):
    # Replace empty or 'unknown' with NaN
    data.replace(['', ' ', 'unknown'], np.nan, inplace=True)
    
    # Fill categorical missing values
    data.loc[:, 'State'] = data['State'].fillna('Unknown')
    data.loc[:, 'International plan'] = data['International plan'].fillna('No')
    data.loc[:, 'Voice mail plan'] = data['Voice mail plan'].fillna('No')
    
    # Fill numerical missing values
    numerical_cols = [
        'Total day minutes', 'Total day calls', 'Total day charge',
        'Total eve minutes', 'Total eve calls', 'Total eve charge',
        'Total night minutes', 'Total night calls', 'Total night charge',
        'Total intl minutes', 'Total intl calls', 'Total intl charge'
    ]
    data.loc[:, numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())
    
    # Fill 'Churn' with mode
    if data['Churn'].isnull().sum() > 0:
        data.loc[:, 'Churn'] = data['Churn'].fillna(data['Churn'].mode()[0])
    
    return data

