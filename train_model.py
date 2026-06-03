"""
train_model.py
--------------
Trains the initial Machine Learning model using RandomForestClassifier.
Splits the labeled dataset, evaluates prediction metrics, and serializes the model.
"""

import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

from preprocess import load_and_preprocess_data
from utils import save_model, log_info, log_success, log_error, log_warn

def train_model(destination=None):
    """
    Load data, perform train-test split, fit a Random Forest Classifier,
    evaluate on test data, and save the model to disk.
    """
    if destination:
        log_info(f"Starting model training pipeline for destination: '{destination}'...")
    else:
        log_info("Starting global model training pipeline...")
    
    # 1. Load and preprocess the labeled dataset
    X, y = load_and_preprocess_data(destination=destination)
    
    if X is None or y is None:
        if destination:
            log_warn(f"Falling back to global model training because no specific data for '{destination}' was found.")
            return train_model(destination=None)
        else:
            log_error("Failed to train model: Labeled dataset is empty or could not be loaded.")
            return False
        
    # Check minimum records threshold
    min_required_records = 10
    if len(X) < min_required_records:
        log_warn(f"Dataset only has {len(X)} records. We recommend at least {min_required_records} to avoid overfitting.")
    
    # 2. Train-Test Split (80% train, 20% test)
    # Handle extremely small datasets gracefully (e.g. if test_size=0.2 results in 0 test records)
    test_size = 0.2
    if len(X) < 5:
        log_warn("Dataset is extremely small. Skipping train-test split and training on the full set.")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() > 1 else None
        )
        
    log_info(f"Split data into Train: {len(X_train)} records, Test: {len(X_test)} records.")
    
    # 3. Initialize and train RandomForestClassifier
    # We set random_state for reproducible results
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=6, 
        random_state=42,
        class_weight="balanced"  # Handles class imbalances if rain is rare
    )
    
    log_info("Fitting RandomForestClassifier...")
    try:
        clf.fit(X_train, y_train)
    except Exception as e:
        log_error(f"Error fitting classifier model: {e}")
        return False
        
    # 4. Model Evaluation
    log_info("Evaluating model performance on test set...")
    try:
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*50)
        print("                 MODEL METRICS")
        print("="*50)
        print(f"Accuracy Score: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["No Rain", "Rain"], zero_division=0))
        
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(f"  Predicted No Rain | Predicted Rain")
        print(f"Actual No Rain:  {cm[0][0]:4d} | {cm[0][1]:4d}")
        if len(cm) > 1:
            print(f"Actual Rain:     {cm[1][0]:4d} | {cm[1][1]:4d}")
        else:
            print(f"Actual Rain:     N/A  | N/A  (No Rain records in test set)")
        print("="*50 + "\n")
        
    except Exception as e:
        log_error(f"Evaluation error: {e}")
        # Continue saving the model anyway if training succeeded
        
    # 5. Save the trained model
    success = save_model(clf, destination=destination)
    if success:
        log_success(f"Training pipeline completed successfully{' for ' + destination if destination else ' (global)'}.")
        return True
    else:
        log_error("Failed to serialize and save the model.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the AI travel rain prediction model.")
    parser.add_argument("--dest", type=str, default=None, help="Destination location to train the model for.")
    args = parser.parse_args()
    
    train_model(destination=args.dest)
