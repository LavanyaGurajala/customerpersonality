"""
Data preprocessing module for customer segmentation.
Handles missing values, encoding, and data cleaning.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Fill numeric missing values with median
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical missing values with mode
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def encode_categorical_features(df, categorical_columns=None):
    """
    Encode categorical features using Label Encoding.
    
    Args:
        df: pandas DataFrame
        categorical_columns: list of column names to encode
    
    Returns:
        Encoded DataFrame and dictionary of encoders
    """
    df = df.copy()
    encoders = {}
    
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    return df, encoders

def estimate_income(age, education_level, avg_spend):
    """
    Estimate income based on age, education, and spending patterns.
    
    Args:
        age: customer age
        education_level: education level (encoded or string)
        avg_spend: average spending
    
    Returns:
        Estimated income
    """
    # Base income by age
    base_income = 2550000 + (age - 18) * 85000
    
    # Education multiplier
    education_multipliers = {
        'High School': 1.0,
        'Bachelor': 1.5,
        'Master': 2.0,
        'PhD': 2.5
    }
    
    if isinstance(education_level, str):
        multiplier = education_multipliers.get(education_level, 1.0)
    else:
        multiplier = 1.0 + (education_level * 0.5)
    
    # Spending factor (people who spend more likely earn more)
    spending_factor = 1 + (avg_spend / 85000)
    
    estimated = base_income * multiplier * spending_factor
    return min(max(estimated, 1700000), 17000000)  # Cap between 17L and 1.7Cr

def clean_dataset(df):
    """
    Complete data cleaning pipeline.
    
    Args:
        df: raw pandas DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Remove outliers using IQR method for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df
