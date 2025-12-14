"""
K-Means clustering model training module.
Trains and saves the customer segmentation model.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

def select_features_for_clustering(df):
    """
    Select relevant features for clustering.
    
    Args:
        df: pandas DataFrame with all features
    
    Returns:
        DataFrame with selected features
    """
    # Key features for customer segmentation
    feature_columns = [
        'Age',
        'Income',
        'TotalSpent',
        'AvgPurchaseValue',
        'EngagementScore',
        'Marital_Status',
        'Education'
    ]
    
    # Select available columns
    available_features = [col for col in feature_columns if col in df.columns]
    
    return df[available_features]

def determine_optimal_clusters(X, max_clusters=10):
    """
    Determine optimal number of clusters using elbow method.
    
    Args:
        X: Feature matrix
        max_clusters: Maximum number of clusters to test
    
    Returns:
        Optimal number of clusters
    """
    inertias = []
    K_range = range(2, min(max_clusters + 1, len(X)))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Simple elbow detection (using rate of change)
    if len(inertias) >= 3:
        differences = np.diff(inertias)
        rate_of_change = np.diff(differences)
        optimal_k = np.argmin(rate_of_change) + 3  # +3 because of double diff
        return min(max(optimal_k, 3), 6)  # Constrain between 3 and 6
    
    return 4  # Default to 4 clusters

def train_kmeans_model(df, n_clusters=None):
    """
    Train K-Means clustering model.
    
    Args:
        df: pandas DataFrame with features
        n_clusters: Number of clusters (None for automatic)
    
    Returns:
        Trained model, scaler, and feature columns
    """
    # Select features
    X = select_features_for_clustering(df)
    feature_columns = X.columns.tolist()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal clusters if not provided
    if n_clusters is None:
        n_clusters = determine_optimal_clusters(X_scaled)
    
    print(f"Training K-Means with {n_clusters} clusters...")
    
    # Train model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_scaled)
    
    # Add cluster labels to dataframe
    df['Cluster'] = kmeans.labels_
    
    return kmeans, scaler, feature_columns, df

def save_model(model, scaler, feature_columns, filepath='models/kmeans_model.pkl'):
    """
    Save trained model, scaler, and feature columns.
    
    Args:
        model: Trained K-Means model
        scaler: Fitted StandardScaler
        feature_columns: List of feature column names
        filepath: Path to save the model
    """
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save all components together
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns
    }
    
    joblib.dump(model_data, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath='models/kmeans_model.pkl'):
    """
    Load trained model, scaler, and feature columns.
    
    Args:
        filepath: Path to saved model
    
    Returns:
        Dictionary with model, scaler, and feature_columns
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model_data = joblib.load(filepath)
    return model_data
