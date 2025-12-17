"""
Utility functions for data loading, transformation, and preprocessing.
Used in the SmartBank Churn Prediction Flask application.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, PowerTransformer, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(demo_file, service_file, activity_file, trans_file):
    """
    Load four CSV files into pandas dataframes.
    
    Parameters:
        demo_file: Customer demographics CSV
        service_file: Customer service interactions CSV
        activity_file: Online activity CSV
        trans_file: Transaction history CSV
    
    Returns:
        Tuple of (demo_df, service_df, activity_df, trans_df)
    """
    demo_df = pd.read_csv(demo_file)
    service_df = pd.read_csv(service_file)
    activity_df = pd.read_csv(activity_file)
    trans_df = pd.read_csv(trans_file)
    
    return demo_df, service_df, activity_df, trans_df


# ============================================================================
# TRANSFORMATIONS
# ============================================================================

def transform_customer_service(customer_service):
    """
    Transform customer service data by grouping by CustomerID.
    
    Steps:
        1. Encode ResolutionStatus as binary (0/1)
        2. Group by CustomerID and aggregate interactions
        3. Replace InteractionDate with DaysSinceLastInteraction
        4. Calculate ResolutionRate
    
    Returns:
        Transformed dataframe with CustomerID as key
    """
    df = customer_service.copy()
    
    # Binarize resolution status
    df['ResolutionStatus'] = df['ResolutionStatus'].replace({'Unresolved': 0, 'Resolved': 1})
    
    # Convert date to datetime
    df['InteractionDate'] = pd.to_datetime(df['InteractionDate'])
    
    # Group by CustomerID
    df = df.groupby('CustomerID').agg({
        'InteractionID': 'count',
        'InteractionDate': 'max',
        'ResolutionStatus': 'sum'
    }).reset_index().rename(columns={
        'InteractionID': 'NumInteractions',
        'ResolutionStatus': 'InteractionsResolved'
    })
    
    # Calculate days since last interaction
    max_date = df['InteractionDate'].max()
    df['DaysSinceLastInteraction'] = (max_date - df['InteractionDate']).dt.days
    
    # Calculate resolution rate
    df['ResolutionRate'] = round(df['InteractionsResolved'] / df['NumInteractions'], 2)
    
    # Drop intermediate columns
    df.drop(columns=['InteractionDate', 'NumInteractions', 'InteractionsResolved'], inplace=True)
    
    return df


# ============================================================================

def transform_online_activity(online_activity):
    """
    Transform online activity data by grouping by CustomerID.
    
    Steps:
        1. One-hot encode ServiceUsage
        2. Group by CustomerID and aggregate
        3. Replace LastLoginDate with DaysSinceLastLogin
    
    Returns:
        Transformed dataframe with CustomerID as key
    """
    df = online_activity.copy()
    
    # One-hot encode service usage
    df = pd.get_dummies(df, columns=['ServiceUsage'], drop_first=True)
    
    # Convert date to datetime
    df['LastLoginDate'] = pd.to_datetime(df['LastLoginDate'])
    
    # Group by CustomerID
    df = df.groupby('CustomerID').agg({
        'LastLoginDate': 'max',
        'LoginFrequency': 'max',
        'ServiceUsage_Online Banking': 'sum',
        'ServiceUsage_Website': 'sum'
    }).reset_index()
    
    # Calculate days since last login
    max_date = df['LastLoginDate'].max()
    df['DaysSinceLastLogin'] = (max_date - df['LastLoginDate']).dt.days
    
    # Drop intermediate columns
    df.drop(columns=['LastLoginDate'], inplace=True)
    
    return df


# ============================================================================

def transform_transaction_history(transaction_history):
    """
    Transform transaction history data by grouping by CustomerID.
    
    Steps:
        1. Group by CustomerID and aggregate transactions
        2. Replace TransactionDate with DaysSinceLastTransaction
    
    Returns:
        Transformed dataframe with CustomerID as key
    """
    df = transaction_history.copy()
    
    # Convert date to datetime
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    
    # Group by CustomerID
    df = df.groupby('CustomerID').agg({
        'TransactionID': 'count',
        'AmountSpent': 'sum',
        'TransactionDate': 'max'
    }).reset_index().rename(columns={'TransactionID': 'NumTransactions'})
    
    # Calculate days since last transaction
    max_date = df['TransactionDate'].max()
    df['DaysSinceLastTransaction'] = (max_date - df['TransactionDate']).dt.days
    
    # Drop intermediate columns
    df.drop(columns=['TransactionDate'], inplace=True)
    
    return df


# ============================================================================
# DATA MERGING
# ============================================================================

def merge_data(customer_demographics, customer_service, online_activity, transaction_history):
    """
    Merge all transformed dataframes on CustomerID.
    
    Parameters:
        customer_demographics: Demographics data
        customer_service: Service data (transformed)
        online_activity: Activity data (transformed)
        transaction_history: Transaction data (transformed)
    
    Returns:
        Merged dataframe with all features
    """
    data = customer_demographics.merge(customer_service, on='CustomerID', how='outer')
    data = data.merge(online_activity, on='CustomerID', how='outer')
    data = data.merge(transaction_history, on='CustomerID', how='outer')
    return data


# ============================================================================
# DATA CLEANING & PREPROCESSING
# ============================================================================

def clean_data(df):
    """
    Handle missing values and data type conversions.
    
    Steps:
        1. Fill NAs in ResolutionRate and DaysSinceLastInteraction with -1
        2. Convert DaysSinceLastInteraction to int64
        3. Drop unnecessary columns (AmountSpent)
    
    Returns:
        Cleaned dataframe
    """
    df = df.copy()
    
    # Fill missing values (-1 indicates no interactions)
    df[['ResolutionRate', 'DaysSinceLastInteraction']] = df[[
        'ResolutionRate', 'DaysSinceLastInteraction'
    ]].fillna({'ResolutionRate': -1, 'DaysSinceLastInteraction': -1})
    
    # Fix data types
    df['DaysSinceLastInteraction'] = df['DaysSinceLastInteraction'].astype('int64')
    
    # Drop multicollinear column
    df = df.drop(columns=['AmountSpent'], errors='ignore')
    
    return df


def engineer_features(df):
    """
    Create derived features from existing ones.
    
    New features:
        - TransactionsPerLogin: Transaction volume relative to login frequency
        - RecencyScore: Average days of inactivity across all touchpoints
        - ActivityMomentum: Login frequency adjusted by recency
    
    Returns:
        Dataframe with engineered features
    """
    df = df.copy()
    
    # Create derived features
    df['TransactionsPerLogin'] = df['NumTransactions'] / (df['LoginFrequency'] + 1)
    
    df['RecencyScore'] = (
        df['DaysSinceLastTransaction'] + 
        df['DaysSinceLastLogin'] + 
        df['DaysSinceLastInteraction']
    ) / 3
    
    df['ActivityMomentum'] = df['LoginFrequency'] / (df['RecencyScore'] + 1)
    
    return df


# ============================================================================
# MODEL LOADING & PREPROCESSING
# ============================================================================

def load_model_and_transformer(model_path='../outputs/lgbm_model.pkl', 
                               transformer_path='../outputs/lgbm_transformer.pkl'):
    """
    Load the trained LightGBM model and fitted transformer.
    
    Returns a sklearn pipeline combining the transformer and model.
    
    Parameters:
        model_path: Path to the saved LightGBM model pickle file
        transformer_path: Path to the saved fitted transformer pickle file
    
    Returns:
        sklearn Pipeline object (transformer + model combined)
    """
    # Load both transformer and model
    model = pickle.load(open(model_path, 'rb'))
    transformer = pickle.load(open(transformer_path, 'rb'))
    
    # Combine into pipeline
    full_pipeline = make_pipeline(transformer, model)
    
    return full_pipeline


