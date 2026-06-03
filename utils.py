"""
utils.py
--------
Utility functions and configuration constants for the Travel Weather Prediction System.
Handles directory paths, Supabase database interactions, model serialization to Supabase Storage, and terminal logging.
Includes helper function to compute historical weather API accuracy.
"""

import os
import sys
import pandas as pd
import joblib
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# We still need a local temp dir for Scikit-Learn to save/load .pkl files before uploading/downloading to Supabase
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_MODEL_DIR = "/tmp/mazha_models"

# CSV Dataset Column Definitions for Travel Destinations
COLUMNS = [
    "timestamp",
    "destination",
    "travel_duration",
    "estimated_arrival_time",
    "destination_temperature",
    "destination_humidity",
    "destination_pressure",
    "destination_wind_speed",
    "destination_cloud_coverage",
    "destination_api_rain_probability",
    "destination_api_prediction",
    "time_of_day",
    "month",
    "historical_prediction_accuracy",
    "model_prediction",
    "model_confidence",
    "actual_rain",
    "prediction_correctness"
]

def ensure_directories():
    """Ensure that the local temp model directory exists."""
    try:
        os.makedirs(TEMP_MODEL_DIR, exist_ok=True)
    except Exception as e:
        log_error(f"Failed to create directories: {e}")
        pass

def log_info(msg):
    print(f"\033[94m[INFO]\033[0m {msg}")

def log_success(msg):
    print(f"\033[92m[SUCCESS]\033[0m {msg}")

def log_warn(msg):
    print(f"\033[93m[WARNING]\033[0m {msg}")

def log_error(msg):
    print(f"\033[91m[ERROR]\033[0m {msg}", file=sys.stderr)

def check_supabase():
    if not supabase:
        log_error("Supabase client not initialized. Check SUPABASE_URL and SUPABASE_KEY in .env")
        return False
    return True

def load_dataset():
    """
    Fetch all records from the Supabase 'weather_data' table and return as a Pandas DataFrame.
    """
    if not check_supabase():
        return pd.DataFrame(columns=COLUMNS)
        
    try:
        response = supabase.table("weather_data").select("*").execute()
        data = response.data
        if not data:
            return pd.DataFrame(columns=COLUMNS)
        df = pd.DataFrame(data)
        # Convert timestamp strings to datetime if needed
        return df
    except Exception as e:
        log_error(f"Error loading dataset from Supabase: {e}")
        return pd.DataFrame(columns=COLUMNS)

def calculate_historical_accuracy(destination_city=None):
    """
    Calculates the historical accuracy of the OWM weather forecast directly from Supabase.
    """
    if not check_supabase():
        return 0.80
        
    try:
        # Fetch labeled data
        response = supabase.table("weather_data").select("destination, destination_api_prediction, actual_rain").not_.is_("actual_rain", "null").execute()
        data = response.data
        if not data:
            return 0.80
            
        df = pd.DataFrame(data)
        if len(df) == 0:
            return 0.80
            
        target_df = df
        if destination_city:
            city_mask = df["destination"].astype(str).str.lower().str.strip() == destination_city.lower().strip()
            if city_mask.sum() >= 3:
                target_df = df[city_mask]
                
        correct_forecasts = (target_df["destination_api_prediction"].astype(int) == target_df["actual_rain"].astype(int)).sum()
        total_forecasts = len(target_df)
        
        accuracy = float(correct_forecasts / total_forecasts)
        return round(accuracy, 4)
    except Exception as e:
        log_warn(f"Failed to calculate historical accuracy from Supabase: {e}. Defaulting to 88%.")
        return 0.80

def append_record(record):
    """
    Inserts a new weather prediction record into Supabase.
    """
    if not check_supabase():
        return False
        
    try:
        if "timestamp" not in record or not record["timestamp"]:
            record["timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            
        # Clean up NaNs or unsupported types for JSON serialization
        for key, value in record.items():
            if pd.isna(value) or value == "":
                record[key] = None
                
        supabase.table("weather_data").insert(record).execute()
        log_info(f"Appended new travel record for '{record.get('destination', 'unknown')}' to Supabase dataset.")
        return True
    except Exception as e:
        log_error(f"Failed to append record to Supabase: {e}")
        return False

def update_latest_feedback(actual_rain):
    """
    Finds the latest entry in Supabase without feedback and updates the 'actual_rain' and 'prediction_correctness' columns.
    """
    if not check_supabase():
        return False
        
    try:
        # Get the most recent record with NULL actual_rain
        response = supabase.table("weather_data").select("id, timestamp, model_prediction").is_("actual_rain", "null").order("timestamp", desc=True).limit(1).execute()
        
        target_id = None
        model_pred = None
        timestamp = None
        
        if response.data and len(response.data) > 0:
            target_id = response.data[0]["id"]
            model_pred = response.data[0]["model_prediction"]
            timestamp = response.data[0]["timestamp"]
            log_info(f"Found pending record without feedback at timestamp: {timestamp}")
        else:
            # If all have feedback, get the absolute latest record
            fallback = supabase.table("weather_data").select("id, timestamp, model_prediction").order("timestamp", desc=True).limit(1).execute()
            if not fallback.data or len(fallback.data) == 0:
                log_warn("No records found in the Supabase dataset to apply feedback to.")
                return False
            target_id = fallback.data[0]["id"]
            model_pred = fallback.data[0]["model_prediction"]
            timestamp = fallback.data[0]["timestamp"]
            log_warn(f"All records have feedback. Overwriting the latest record at: {timestamp}")
            
        # Calculate correctness
        correctness = 1 if int(model_pred) == int(actual_rain) else 0
        
        # Update row
        update_data = {
            "actual_rain": int(actual_rain),
            "prediction_correctness": correctness
        }
        supabase.table("weather_data").update(update_data).eq("id", target_id).execute()
        
        log_success(f"Updated record at {timestamp} in Supabase with actual_rain = {actual_rain}, correctness = {correctness}.")
        return True
    except Exception as e:
        log_error(f"Failed to update feedback in Supabase: {e}")
        return False

def save_model(model, destination=None):
    """
    Serialize the trained model locally to /tmp, then upload it to the Supabase 'models' bucket.
    """
    if not check_supabase():
        return False
        
    ensure_directories()
    try:
        model_filename = "rain_model.pkl"
        if destination:
            safe_dest = str(destination).lower().strip().replace(" ", "_")
            model_filename = f"rain_model_{safe_dest}.pkl"
            
        local_path = os.path.join(TEMP_MODEL_DIR, model_filename)
        
        # Save to local temporary path
        joblib.dump(model, local_path)
        
        # Upload to Supabase Storage (overwrite if exists)
        with open(local_path, "rb") as f:
            supabase.storage.from_("models").upload(
                path=model_filename,
                file=f,
                file_options={"cache-control": "3600", "upsert": "true"}
            )
            
        log_success(f"Model saved successfully to Supabase Storage: {model_filename}")
        return True
    except Exception as e:
        log_error(f"Failed to save model to Supabase: {e}")
        return False

def load_model(destination=None):
    """
    Download the model from the Supabase 'models' bucket to /tmp, then load it.
    """
    if not check_supabase():
        return None
        
    ensure_directories()
    model_filename = "rain_model.pkl"
    if destination:
        safe_dest = str(destination).lower().strip().replace(" ", "_")
        model_filename = f"rain_model_{safe_dest}.pkl"
        
    local_path = os.path.join(TEMP_MODEL_DIR, model_filename)
    
    try:
        # Download from Supabase
        with open(local_path, "wb+") as f:
            res = supabase.storage.from_("models").download(model_filename)
            f.write(res)
            
        # Load the model
        model = joblib.load(local_path)
        return model
    except Exception as e:
        # E.g., if the file doesn't exist in the bucket yet
        log_warn(f"No trained model found in Supabase Storage for {model_filename} (or download failed: {e})")
        return None

def get_all_destinations():
    """
    Fetch a list of all unique destination locations from Supabase.
    """
    if not check_supabase():
        return []
        
    try:
        response = supabase.table("weather_data").select("destination").execute()
        if not response.data:
            return []
            
        df = pd.DataFrame(response.data)
        destinations = df["destination"].dropna().unique().tolist()
        return [str(d).strip() for d in destinations if str(d).strip()]
    except Exception as e:
        log_warn(f"Failed to fetch destinations from Supabase: {e}")
        return []


def update_trip_feedback(timestamp, feedback_map):
    if not check_supabase():
        return 0
        
    try:
        response = supabase.table("weather_data").select("id, destination, model_prediction").eq("timestamp", timestamp).execute()
        print(f"DB Rows found for timestamp {timestamp}: {response.data}")
        if not response.data:
            return 0
            
        updated_count = 0
        for row in response.data:
            dest = str(row["destination"]).lower()
            if dest in feedback_map:
                actual_rain = int(feedback_map[dest])
                model_pred = int(row["model_prediction"])
                correctness = 1 if model_pred == actual_rain else 0
                
                supabase.table("weather_data").update({
                    "actual_rain": actual_rain,
                    "prediction_correctness": correctness
                }).eq("id", row["id"]).execute()
                
                updated_count += 1
                
        return updated_count
    except Exception as e:
        log_error(f"Failed to update trip feedback in Supabase: {e}")
        return 0
