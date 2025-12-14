"""
Script to run the data preprocessing pipeline.
Reads raw data, cleans it, encodes categorical features, and saves the result.
"""
import pandas as pd
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.preprocessing import clean_dataset, encode_categorical_features

def main():
    # Define paths
    data_dir = os.path.join(parent_dir, 'data')
    input_path = os.path.join(data_dir, 'marketing_campaign.csv')
    output_path = os.path.join(data_dir, 'preprocessed_data.csv')
    
    print("Starting data preprocessing...")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return
        
    # Load dataset
    # Try reading with different separators
    try:
        df = pd.read_csv(input_path, sep='\t')
        if df.shape[1] < 2:  # If tab didn't work, try comma
             df = pd.read_csv(input_path, sep=',')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Loaded dataset with shape: {df.shape}")
    

    # Convert Currency to INR (x85)
    print("Converting currency to INR (x85)...")
    currency_cols = ['Income'] + [col for col in df.columns if 'Mnt' in col]
    for col in currency_cols:
        if col in df.columns:
            df[col] = df[col] * 85
            
    # Data Cleaning
    print("Cleaning dataset...")
    df_cleaned = clean_dataset(df)
    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    print(f"Removed {len(df) - len(df_cleaned)} rows")
    
    # Encode Categorical Features
    print("Encoding categorical features...")
    categorical_cols = ['Marital_Status', 'Education']
    df_encoded, encoders = encode_categorical_features(df_cleaned, categorical_cols)
    
    # Print encoding mappings for verification
    print("\nEncoding mappings:")
    for col, encoder in encoders.items():
        print(f"\n{col}:")
        for i, label in enumerate(encoder.classes_):
            print(f"  {label} -> {i}")
            
    # Save preprocessed data
    print(f"\nSaving preprocessed data to {output_path}...")
    df_encoded.to_csv(output_path, index=False)
    print("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()
