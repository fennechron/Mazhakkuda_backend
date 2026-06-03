"""
preprocess.py
-------------
Preprocesses the weather dataset for model training and inference.
Responsible for converting categorical location strings to numerical coordinates/IDs,
extracting time-of-day/month features, and formatting datasets for the Scikit-learn model.
"""

import pandas as pd
import numpy as np
from utils import load_dataset, log_info, log_warn, log_error

# Feature names used for training and prediction
FEATURE_COLS = [
    "destination",
    "travel_duration",
    "destination_temperature",
    "destination_humidity",
    "destination_pressure",
    "destination_wind_speed",
    "destination_cloud_coverage",
    "destination_api_rain_probability",
    "time_of_day",
    "month",
    "historical_prediction_accuracy"
]
TARGET_COL = "actual_rain"

def encode_location(location_name):
    """
    Deterministic string-to-integer hashing to convert location names into numeric features.
    Provides a stable numeric encoding without requiring stateful fitting (LabelEncoder),
    preventing key errors when predicting on new/unseen cities.
    """
    if pd.isna(location_name) or not isinstance(location_name, str) or not location_name.strip():
        return 0
    
    # Process string to be case and space insensitive
    clean_name = location_name.lower().strip()
    
    # Simple polynomial rolling hash representation
    char_sum = sum(ord(char) * (idx + 1) for idx, char in enumerate(clean_name))
    return (char_sum % 999) + 1  # maps to range [1, 999]

def load_and_preprocess_data(destination=None):
    """
    Loads travel weather dataset, cleans it, filters out rows missing ground truth (actual_rain),
    encodes location strings, and returns features X and target labels y.
    
    Returns:
    tuple: (X, y) where X is a pandas DataFrame and y is a pandas Series, or (None, None) on failure.
    """
    try:
        df = load_dataset()
        
        if df.empty:
            log_warn("The dataset is empty. Cannot preprocess.")
            return None, None
            
        if destination:
            safe_dest = str(destination).lower().strip()
            df = df[df["destination"].astype(str).str.lower().str.strip() == safe_dest]
            
            if df.empty:
                log_warn(f"No records found in the dataset for destination '{destination}'.")
                return None, None
                
        # Filter where actual_rain has feedback
        labeled_df = df.copy()
        labeled_df[TARGET_COL] = pd.to_numeric(labeled_df[TARGET_COL], errors='coerce')
        labeled_df = labeled_df[labeled_df[TARGET_COL].isin([0, 1])]
        
        if len(labeled_df) == 0:
            log_warn("No labeled historical data found (actual_rain is empty/NaN for all rows).")
            return None, None
            
        # Separate features and target
        X = pd.DataFrame(index=labeled_df.index)
        
        # Apply deterministic location encoding
        X["destination"] = labeled_df["destination"].apply(encode_location)
        
        # Process numeric features
        numeric_cols = [
            "travel_duration",
            "destination_temperature",
            "destination_humidity",
            "destination_pressure",
            "destination_wind_speed",
            "destination_cloud_coverage",
            "destination_api_rain_probability",
            "time_of_day",
            "month",
            "historical_prediction_accuracy"
        ]
        
        # Safe default values if columns contain missing data
        default_defaults = {
            "travel_duration": 20.0,
            "destination_temperature": 20.0,
            "destination_humidity": 60,
            "destination_pressure": 1013,
            "destination_wind_speed": 4.0,
            "destination_cloud_coverage": 40,
            "destination_api_rain_probability": 0.2,
            "time_of_day": 12,
            "month": 6,
            "historical_prediction_accuracy": 0.80
        }
        
        for col in numeric_cols:
            vals = pd.to_numeric(labeled_df[col], errors='coerce')
            median_val = vals.median()
            if pd.isna(median_val):
                median_val = default_defaults[col]
            X[col] = vals.fillna(median_val)
            
        y = labeled_df[TARGET_COL].astype(int)
        
        # Double check alignment of column ordering
        X = X[FEATURE_COLS]
        
        log_info(f"Loaded and preprocessed {len(X)} labeled records for model training.")
        return X, y
        
    except Exception as e:
        log_error(f"Preprocessing error: {e}")
        return None, None

def prepare_prediction_features(weather_record):
    """
    Converts a single travel prediction record (dictionary) into a 1-row DataFrame 
    with location mapping and features aligned for inference.
    
    Parameters:
    weather_record (dict): Single record matching the required travel keys.
    
    Returns:
    pandas.DataFrame: A 1-row DataFrame containing the ordered feature values.
    """
    # Create single-row DataFrame
    record_copy = weather_record.copy()
    
    # Encode categorical location strings
    record_copy["destination"] = encode_location(record_copy.get("destination", ""))
    
    # Defaults dictionary
    defaults = {
        "destination": 0,
        "travel_duration": 20.0,
        "destination_temperature": 20.0,
        "destination_humidity": 60,
        "destination_pressure": 1013,
        "destination_wind_speed": 4.0,
        "destination_cloud_coverage": 40,
        "destination_api_rain_probability": 0.2,
        "time_of_day": 12,
        "month": 6,
        "historical_prediction_accuracy": 0.80
    }
    
    # Build single row matching FEATURE_COLS order
    row_data = {}
    for col in FEATURE_COLS:
        val = record_copy.get(col, None)
        if val is None or pd.isna(val):
            row_data[col] = defaults[col]
        else:
            row_data[col] = val
            
    df = pd.DataFrame([row_data])
    
    # Coerce features to numerical types
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col])
        
    # Order features
    return df[FEATURE_COLS]

if __name__ == "__main__":
    print("--- Running standalone preprocessor test ---")
    X, y = load_and_preprocess_data()
    if X is not None:
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print("Feature columns order:", list(X.columns))
        print(X.head(3))
