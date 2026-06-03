"""
save_feedback.py
----------------
Logs user feedback (ground truth) on whether it actually rained at the destination upon arrival.
Updates the latest pending prediction record in the Supabase dataset with the user's feedback.
"""

import sys
import argparse
from utils import load_dataset, log_info, log_success, log_error, log_warn, check_supabase, supabase

def parse_arguments():
    parser = argparse.ArgumentParser(description="Save weather feedback (ground truth labels) for travel rain prediction")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rain", action="store_true", help="Record that it did rain")
    group.add_argument("--no-rain", action="store_true", help="Record that it did not rain")
    return parser.parse_args()

def run_feedback():
    args = parse_arguments()
    if not check_supabase():
        log_error("Supabase not configured. Cannot save feedback.")
        sys.exit(1)
        
    df = load_dataset()
    if len(df) == 0:
        log_error("The dataset is empty.")
        sys.exit(1)
        
    unlabeled_mask = df["actual_rain"].isna()
    if not unlabeled_mask.any():
        log_warn("No pending feedback found.")
        sys.exit(0)
        
    pending_timestamps = df[unlabeled_mask]["timestamp"].unique()
    latest_ts = pending_timestamps[-1]
    
    subset = df[df["timestamp"] == latest_ts]
    records = subset.to_dict(orient="records")
    
    for record in records:
        dest = record["destination"]
        print(f"Checkpoint: {dest}")
        actual_rain = None
        if args.rain:
            actual_rain = 1
        elif args.no_rain:
            actual_rain = 0
        else:
            user_input = input(f"Did it actually rain at {dest}? (1 = Rain, 0 = No Rain, s = Skip): ").strip().lower()
            if user_input in ['1', 'r', 'y']:
                actual_rain = 1
            elif user_input in ['0', 'n', 'nr']:
                actual_rain = 0
                
        if actual_rain is not None:
            model_pred = int(record["model_prediction"])
            correctness = 1 if model_pred == actual_rain else 0
            
            supabase.table("weather_data").update({
                "actual_rain": actual_rain,
                "prediction_correctness": correctness
            }).eq("id", record["id"]).execute()
            
            log_success(f"Feedback saved for {dest}.")

if __name__ == "__main__":
    run_feedback()
