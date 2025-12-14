"""
Script to train the K-Means clustering model.
Loads preprocessed data (or raw data), engineers features, trains the model, and saves it.
"""
import pandas as pd
import os
import sys
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.feature_engineering import engineer_features
from src.clustering_model import train_kmeans_model, save_model
from src.preprocessing import clean_dataset

def main():
    # Define paths
    data_dir = os.path.join(parent_dir, 'data')
    models_dir = os.path.join(parent_dir, 'models')
    
    # Prefer checking for preprocessed data first, but robustly fallback to raw
    preprocessed_path = os.path.join(data_dir, 'preprocessed_data.csv')
    raw_path = os.path.join(data_dir, 'marketing_campaign.csv')
    model_output_path = os.path.join(models_dir, 'kmeans_model.pkl')
    
    print("Starting model training pipeline...")
    
    df = None
    
    # Try loading preprocessed data
    if os.path.exists(preprocessed_path):
        print(f"Loading preprocessed data from {preprocessed_path}...")
        df = pd.read_csv(preprocessed_path)
    # If not found, load raw and preprocess
    elif os.path.exists(raw_path):
        print(f"Preprocessed data not found. Loading raw data from {raw_path}...")
        try:
             df = pd.read_csv(raw_path, sep='\t')
             if df.shape[1] < 2:
                 df = pd.read_csv(raw_path, sep=',')
             df = clean_dataset(df)
        except Exception as e:
            print(f"Error processing raw data: {e}")
            return
    else:
        print(f"Error: No data found in {data_dir}")
        return

    print(f"Dataset shape: {df.shape}")
    
    # Feature Engineering
    print("Engineering features...")
    df_features = engineer_features(df)
    
    # Ensure categorical columns are encoded if they aren't already
    # (Preprocessed data might already have them, but feature engineering might rely on strings or re-generations)
    # Ideally, feature engineering uses the raw-ish columns.
    # Let's simple check if they are numeric.
    
    if df_features['Marital_Status'].dtype == 'object':
         print("Encoding Marital_Status...")
         df_features['Marital_Status'] = LabelEncoder().fit_transform(df_features['Marital_Status'].astype(str))
    
    if df_features['Education'].dtype == 'object':
         print("Encoding Education...")
         df_features['Education'] = LabelEncoder().fit_transform(df_features['Education'].astype(str))

    
    # Train Model
    print("Training K-Means model...")
    # Using 5 clusters for finer granularity
    model, scaler, feature_cols, df_clustered = train_kmeans_model(df_features, n_clusters=5)
    
    print("\nCluster distribution:")
    print(df_clustered['Cluster'].value_counts().sort_index())
    
    # Save Model
    print(f"\nSaving model to {model_output_path}...")
    save_model(model, scaler, feature_cols, model_output_path)
    print("Model training completed successfully!")

if __name__ == "__main__":
    main()
