"""
Feature engineering module for customer segmentation.
Creates derived features for clustering analysis.
"""
import pandas as pd
import numpy as np

def calculate_total_spend(df, spend_columns):
    """
    Calculate total spending across all product categories.
    
    Args:
        df: pandas DataFrame
        spend_columns: list of column names containing spending data
    
    Returns:
        Series with total spending
    """
    return df[spend_columns].sum(axis=1)

def calculate_average_purchase_value(total_spend, num_purchases):
    """
    Calculate average value per purchase.
    
    Args:
        total_spend: total amount spent
        num_purchases: number of purchases
    
    Returns:
        Average purchase value
    """
    return np.where(num_purchases > 0, total_spend / num_purchases, 0)

def calculate_engagement_score(df):
    """
    Calculate customer engagement score based on various factors.
    
    Args:
        df: pandas DataFrame with customer data
    
    Returns:
        Series with engagement scores
    """
    # Normalize web visits and catalog purchases
    web_visits_norm = (df['NumWebVisitsMonth'] - df['NumWebVisitsMonth'].min()) / \
                      (df['NumWebVisitsMonth'].max() - df['NumWebVisitsMonth'].min() + 1e-10)
    
    catalog_norm = (df['NumCatalogPurchases'] - df['NumCatalogPurchases'].min()) / \
                   (df['NumCatalogPurchases'].max() - df['NumCatalogPurchases'].min() + 1e-10)
    
    # Weighted engagement score
    engagement = (web_visits_norm * 0.4) + (catalog_norm * 0.3) + \
                 (df['NumStorePurchases'] / df['NumStorePurchases'].max() * 0.3)
    
    return engagement * 100  # Scale to 0-100

def create_age_groups(age):
    """
    Categorize age into groups.
    
    Args:
        age: customer age
    
    Returns:
        Age group category
    """
    if age < 30:
        return 'Young Adult (18-29)'
    elif age < 45:
        return 'Adult (30-44)'
    elif age < 60:
        return 'Middle-aged (45-59)'
    else:
        return 'Senior (60+)'

def create_income_groups(income):
    """
    Categorize income into ranges.
    
    Args:
        income: customer income
    
    Returns:
        Income range category
    """
    if income < 2550000:
        return 'Low Income (<₹25.5L)'
    elif income < 5100000:
        return 'Middle Income (₹25.5L-₹51.0L)'
    elif income < 7650000:
        return 'Upper-Middle Income (₹51.0L-₹76.5L)'
    else:
        return 'High Income (₹76.5L+)'

def identify_preferred_products(df):
    """
    Identify preferred product categories based on spending.
    
    Args:
        df: DataFrame with spending columns
    
    Returns:
        Series with preferred product categories
    """
    product_columns = [col for col in df.columns if 'Mnt' in col]
    
    if not product_columns:
        return pd.Series(['General Products'] * len(df))
    
    preferred = df[product_columns].idxmax(axis=1)
    # Clean column names to readable format
    preferred = preferred.str.replace('Mnt', '').str.replace('Products', '')
    
    return preferred

def engineer_features(df):
    """
    Complete feature engineering pipeline.
    
    Args:
        df: pandas DataFrame with raw customer data
    
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Calculate total spending
    spend_columns = [col for col in df.columns if 'Mnt' in col]
    if spend_columns:
        df['TotalSpent'] = calculate_total_spend(df, spend_columns)
    
    # Calculate average purchase value
    if 'TotalSpent' in df.columns and 'NumStorePurchases' in df.columns:
        total_purchases = df['NumStorePurchases'] + \
                         df.get('NumWebPurchases', 0) + \
                         df.get('NumCatalogPurchases', 0)
        df['AvgPurchaseValue'] = calculate_average_purchase_value(
            df['TotalSpent'], total_purchases
        )
    
    # Calculate engagement score
    if 'NumWebVisitsMonth' in df.columns:
        df['EngagementScore'] = calculate_engagement_score(df)
    
    return df
