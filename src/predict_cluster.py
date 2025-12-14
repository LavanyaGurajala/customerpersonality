"""
Prediction module for customer segmentation.
Predicts cluster and generates business insights.
"""
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.clustering_model import load_model
from src.preprocessing import estimate_income
from src.feature_engineering import create_age_groups, create_income_groups

def map_cluster_to_segment(cluster_id, customer_data):
    """
    Map cluster ID to business-friendly segment with detailed insights.
    
    Args:
        cluster_id: Cluster identifier
        customer_data: Dictionary with customer information
    
    Returns:
        Dictionary with segment insights
    """
    age = customer_data['age']
    income = customer_data['income']
    total_spent = customer_data['total_spent']
    avg_purchase = customer_data['avg_purchase']
    
    # Define segment characteristics
    segments = {
        0: {
            'name': 'Premium High-Value Customers',
            'spending_pattern': 'High frequency, premium purchases with above-average transaction values',
            'preferred_products': 'Luxury items, premium brands, exclusive collections',
            'marketing_action': 'Offer VIP rewards, exclusive early access, personalized premium experiences'
        },
        1: {
            'name': 'Budget-Conscious Shoppers',
            'spending_pattern': 'Price-sensitive with moderate purchase frequency, seeks value deals',
            'preferred_products': 'Discounted items, bundle offers, essential products',
            'marketing_action': 'Target with promotional offers, loyalty discounts, seasonal sales'
        },
        2: {
            'name': 'Occasional Buyers',
            'spending_pattern': 'Low engagement with infrequent purchases, needs activation',
            'preferred_products': 'Basic necessities, impulse purchases',
            'marketing_action': 'Re-engagement campaigns, special incentives, reminder communications'
        },
        3: {
            'name': 'Growing Potential Customers',
            'spending_pattern': 'Moderate spending with increasing purchase frequency trend',
            'preferred_products': 'Mix of essential and premium items, experimenting with categories',
            'marketing_action': 'Nurture with personalized recommendations, cross-sell opportunities'
        },
        4: {
            'name': 'Consistent Regular Customers',
            'spending_pattern': 'Stable, predictable purchase patterns with good retention',
            'preferred_products': 'Preferred brands, routine purchases, subscription items',
            'marketing_action': 'Maintain satisfaction, introduce new products, subscription programs'
        }
    }
    
    # Removed hardcoded overrides to allow the K-Means model to drive segmentation
    # The model provides better multi-dimensional analysis than simple threshold rules.
    
    segment = segments.get(cluster_id % 5, segments[0])
    
    return {
        'segment_name': segment['name'],
        'spending_pattern': segment['spending_pattern'],
        'preferred_products': segment['preferred_products'],
        'age_group': create_age_groups(age),
        'income_range': create_income_groups(income),
        'marketing_action': segment['marketing_action']
    }

def prepare_input_features(age, income, marital_status, education, 
                          total_spent, num_purchases):
    """
    Prepare input features for prediction.
    
    Args:
        age: Customer age
        income: Customer income (can be None)
        marital_status: Marital status
        education: Education level
        total_spent: Total amount spent
        num_purchases: Number of purchases
    
    Returns:
        DataFrame with prepared features
    """
    # Estimate income if not provided
    if income is None:
        avg_purchase = total_spent / num_purchases if num_purchases > 0 else 0
        income = estimate_income(age, education, avg_purchase)
    
    # Calculate derived features
    avg_purchase_value = total_spent / num_purchases if num_purchases > 0 else 0
    engagement_score = min((num_purchases / 12) * 30, 100)  # Simple engagement calculation
    
    # Encode categorical variables
    marital_status_encoding = {
        'Single': 0, 'Married': 1, 'Divorced': 2, 'Widowed': 3
    }
    education_encoding = {
        'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3
    }
    
    features = {
        'Age': age,
        'Income': income,
        'TotalSpent': total_spent,
        'AvgPurchaseValue': avg_purchase_value,
        'EngagementScore': engagement_score,
        'Marital_Status': marital_status_encoding.get(marital_status, 0),
        'Education': education_encoding.get(education, 1)
    }
    
    return pd.DataFrame([features]), income

def predict_customer_segment(age, income, marital_status, education, 
                             total_spent, num_purchases):
    """
    Predict customer segment and return detailed insights.
    
    Args:
        age: Customer age
        income: Customer income (optional)
        marital_status: Marital status
        education: Education level
        total_spent: Total spending
        num_purchases: Number of purchases
    
    Returns:
        Dictionary with segment insights
    """
    try:
        # Load model
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'kmeans_model.pkl')
        
        # If model doesn't exist, create a simple rule-based prediction
        if not os.path.exists(model_path):
            print("Model not found, using rule-based prediction...")
            
            # Prepare features
            features_df, estimated_income = prepare_input_features(
                age, income, marital_status, education, total_spent, num_purchases
            )
            
            # Simple rule-based cluster assignment
            avg_purchase = total_spent / num_purchases if num_purchases > 0 else 0
            
            if total_spent > 127500 and avg_purchase > 8500:
                cluster = 0  # Premium
            elif total_spent < 25500:
                cluster = 2  # Occasional
            elif num_purchases > 15:
                cluster = 4  # Consistent
            elif avg_purchase > 6800:
                cluster = 3  # Growing
            else:
                cluster = 1  # Budget-conscious
            
            customer_data = {
                'age': age,
                'income': estimated_income,
                'total_spent': total_spent,
                'avg_purchase': avg_purchase
            }
            
            return map_cluster_to_segment(cluster, customer_data)
        
        # Load trained model
        model_data = load_model(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        feature_columns = model_data['feature_columns']
        
        # Prepare features
        features_df, estimated_income = prepare_input_features(
            age, income, marital_status, education, total_spent, num_purchases
        )
        
        # Ensure correct column order
        features_df = features_df[feature_columns]
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Predict cluster
        cluster = model.predict(features_scaled)[0]
        
        # Prepare customer data for mapping
        customer_data = {
            'age': age,
            'income': estimated_income,
            'total_spent': total_spent,
            'avg_purchase': total_spent / num_purchases if num_purchases > 0 else 0
        }
        
        # Get segment insights
        return map_cluster_to_segment(cluster, customer_data)
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise
