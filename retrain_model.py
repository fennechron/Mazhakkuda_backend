"""
retrain_model.py
----------------
Continuous learning script that retrains the Random Forest model on the updated dataset.
Imports new labeled feedback data, trains a candidate model, compares its performance 
against the current active model, and replaces it if validation checks pass.
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import sys
import argparse

from preprocess import load_and_preprocess_data
from utils import load_model, save_model, log_info, log_success, log_error, log_warn, get_all_destinations

def evaluate_predictions(y_true, y_pred):
    """Calculate standard metrics for comparison."""
    accuracy = accuracy_score(y_true, y_pred)
    # Use zero_division to handle cases where there are no positive predictions
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def run_retraining(destination=None):
    """Retrain the model on the updated dataset and compare with the existing model."""
    if destination:
        log_info(f"Starting model retraining process for destination '{destination}'...")
    else:
        log_info("Starting global model retraining process...")
    
    # 1. Load updated dataset
    X, y = load_and_preprocess_data(destination=destination)
    if X is None or y is None:
        log_error(f"Retraining aborted: Labeled training dataset could not be loaded{' for ' + destination if destination else ''}.")
        # Don't sys.exit(1) here so the loop can continue for other destinations
        return False
        
    log_info(f"Total labeled records available: {len(X)}")
    
    # 2. Load existing active model (if any) to serve as a baseline
    old_model = load_model(destination=destination)
    
    # 3. Train-Test Split (80% train, 20% test)
    # We use a stratified split to keep the target distributions balanced
    test_size = 0.2
    if len(X) < 5:
        log_warn("Dataset is too small for validation split. Retraining on all available records.")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() > 1 else None
        )
        
    # 4. Train the new candidate model
    new_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        class_weight="balanced"
    )
    
    log_info("Fitting new candidate model...")
    try:
        new_model.fit(X_train, y_train)
    except Exception as e:
        log_error(f"Error training candidate model: {e}")
        sys.exit(1)
        
    # 5. Compare performance
    y_pred_new = new_model.predict(X_test)
    new_metrics = evaluate_predictions(y_test, y_pred_new)
    
    old_metrics = None
    if old_model is not None:
        try:
            y_pred_old = old_model.predict(X_test)
            old_metrics = evaluate_predictions(y_test, y_pred_old)
        except Exception as e:
            log_warn(f"Failed to evaluate old model (it might have a different feature shape): {e}")
            
    # 6. Display Comparison Report
    print("\n" + "="*55)
    print("             RETRAINING COMPARISON REPORT")
    print("="*55)
    
    if old_metrics:
        print(f"Metric      |  Previous Model  |  New Model  |  Difference")
        print("-" * 55)
        for metric_name in ["accuracy", "precision", "recall", "f1_score"]:
            val_old = old_metrics[metric_name]
            val_new = new_metrics[metric_name]
            diff = val_new - val_old
            sign = "+" if diff >= 0 else ""
            print(f"{metric_name.capitalize():11s} |      {val_old:.4f}      |    {val_new:.4f}   |   {sign}{diff:.4f}")
    else:
        print("No previous model metric baseline available for validation.")
        print(f"New Model Accuracy:  {new_metrics['accuracy']:.4f}")
        print(f"New Model Precision: {new_metrics['precision']:.4f}")
        print(f"New Model Recall:    {new_metrics['recall']:.4f}")
        print(f"New Model F1-Score:  {new_metrics['f1_score']:.4f}")
        
    print("="*55 + "\n")
    
    # 7. Safe replacement check: Ensure the new model doesn't severely regress
    # (Since we are using very small datasets, we allow replacements unless accuracy is extremely poor)
    accuracy_threshold = 0.50
    if new_metrics["accuracy"] < accuracy_threshold:
        log_error(f"New candidate model accuracy ({new_metrics['accuracy']:.2%}) is below threshold ({accuracy_threshold:.0%}). Retraining rejected.")
        return False
        
    # 8. Overwrite and activate the new model
    success = save_model(new_model, destination=destination)
    if success:
        log_success(f"New model successfully deployed and activated for future predictions{' (' + destination + ')' if destination else ''}!")
        return True
    else:
        log_error("Could not save retrained model.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain the AI travel rain prediction model.")
    parser.add_argument("--dest", type=str, default=None, help="Specific destination to retrain the model for.")
    args = parser.parse_args()
    
    if args.dest:
        run_retraining(destination=args.dest)
    else:
        destinations = get_all_destinations()
        if not destinations:
            log_info("No known destinations found in dataset. Running global retraining...")
            run_retraining(destination=None)
        else:
            log_info(f"Found {len(destinations)} destinations. Retraining models for all of them...")
            for dest in destinations:
                run_retraining(destination=dest)
            
            # Also train a fallback global model
            log_info("Retraining fallback global model...")
            run_retraining(destination=None)
