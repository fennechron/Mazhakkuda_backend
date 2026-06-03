"""
predict_model.py
----------------
Uses the trained Random Forest model to predict weather conditions at a target travel destination
at the estimated time of arrival.
Integrates routing duration, destination forecast alignment, and dynamic historical accuracy calculation.
"""

import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

from utils import load_model, append_record, log_info, log_warn, log_error, log_success, calculate_historical_accuracy
from travel_time_estimator import estimate_travel_metrics
from route_service import geocode_location, get_driving_route, get_multi_driving_route
from preprocess import prepare_prediction_features
from train_model import train_model

from route_checkpoints import get_route_checkpoints
from eta_calculator import calculate_checkpoints_eta, haversine_distance
from route_weather_forecast import fetch_checkpoint_forecast
from travel_recommendation import generate_route_recommendation

def parse_arguments():
    """Parse command line arguments for travel rain prediction."""
    parser = argparse.ArgumentParser(description="AI Route-Based Weather Prediction & Safety System")
    parser.add_argument(
        "--start", 
        type=str, 
        default=None, 
        help="Starting location/city (default: Kumbanad)"
    )
    parser.add_argument(
        "--dest", 
        type=str, 
        default=None, 
        help="Destination location/city (default: Kochi)"
    )
    parser.add_argument(
        "--duration", 
        type=float, 
        help="Manual travel duration in minutes (overrides route estimator calculation)"
    )
    parser.add_argument(
        "--simulate", 
        action="store_true", 
        help="Force simulated travel routing and weather forecasts"
    )
    parser.add_argument(
        "--api-key", 
        type=str, 
        help="OpenWeatherMap API key (overrides environment variable)"
    )
    parser.add_argument(
        "--route-mode",
        type=str,
        choices=["auto", "custom"],
        default="auto",
        help="Route mode: 'auto' (automatic checkpoints) or 'custom' (uses middle waypoints)"
    )
    parser.add_argument(
        "--waypoints",
        type=str,
        default="",
        help="Comma-separated list of intermediate places (for customized route)"
    )
    return parser.parse_args()

def load_or_train_checkpoint_model(checkpoint_name):
    """Loads model for a checkpoint, training it on-the-fly if missing."""
    model = load_model(destination=checkpoint_name)
    if model is None:
        log_warn(f"Trained model not found for checkpoint '{checkpoint_name}'. Training model automatically...")
        success = train_model(destination=checkpoint_name)
        if success:
            model = load_model(destination=checkpoint_name)
        if model is None:
            log_warn(f"Failed to train model specifically for '{checkpoint_name}'. Falling back to global model...")
            model = load_model(destination=None)
            if model is None:
                log_warn("Global model not found. Training global model...")
                global_success = train_model(destination=None)
                if not global_success:
                    log_error("Could not train global model. Exiting.")
                    sys.exit(1)
                model = load_model(destination=None)
    return model

def geocode_location_with_fallback(name, force_sim=False):
    """Geocodes a location name, with simulation/mock coordinate fallback."""
    name_clean = name.strip().lower()
    if force_sim:
        MOCK_COORDINATES = {
            "kumbanad": (9.3920, 76.6341),
            "chengannur": (9.3175, 76.6111),
            "kochi": (9.9312, 76.2673),
            "thiruvalla": (9.3828, 76.5724),
            "mavelikara": (9.2435, 76.5492),
            "haripad": (9.2847, 76.4357),
            "alappuzha": (9.4981, 76.3388),
            "kozhencherry": (9.3414, 76.6994),
            "pandalam": (9.2272, 76.6853),
            "adoor": (9.1534, 76.7329),
            "kottarakkara": (8.9982, 76.7797),
            "trivandrum": (8.5241, 76.9366),
            "eraviperoor": (9.3828, 76.6379),
            "vallamkulam": (9.3848, 76.6142),
            "kallissery": (9.3400, 76.6150),
            "muthoor": (9.3950, 76.5900)
        }
        if name_clean in MOCK_COORDINATES:
            return MOCK_COORDINATES[name_clean]
        # Hash-based deterministic coordinate generation in Kerala region
        import hashlib
        h = int(hashlib.md5(name_clean.encode()).hexdigest(), 16)
        lat_offset = (h % 100) / 1000.0
        lon_offset = ((h // 100) % 100) / 1000.0
        return (9.35 + lat_offset, 76.6 + lon_offset)
        
    lat, lon = geocode_location(name)
    if lat is None or lon is None:
        return geocode_location_with_fallback(name, force_sim=True)
    return lat, lon

def run_prediction():
    """Main pipeline execution for route-based weather intelligence."""
    args = parse_arguments()
    
    # Check if we should prompt interactively
    if args.start is None and args.dest is None:
        print("\n" + "="*60)
        print("     AI ROUTE WEATHER INTELLIGENCE ASSISTANT - INTERACTIVE")
        print("="*60)
        print("Select Routing Mode:")
        print("  1. Automatic Route (Direct path with auto-sampled checkpoints)")
        print("  2. Customized Route (Specify your own in-between waypoints)")
        mode_choice = input("Enter choice (1 or 2, default 1): ").strip()
        
        if mode_choice == "2":
            args.route_mode = "custom"
            start_city = (input("Enter starting location (Default: kumbanad): ").strip() or "kumbanad").lower()
            waypoints_input = input("Enter intermediate places separated by commas (e.g. thiruvalla, alappuzha): ").strip()
            args.waypoints = waypoints_input
            dest_city = (input("Enter destination location (Default: kochi): ").strip() or "kochi").lower()
        else:
            args.route_mode = "auto"
            start_city = (input("Enter starting location (Default: kumbanad): ").strip() or "kumbanad").lower()
            dest_city = (input("Enter destination location (Default: kochi): ").strip() or "kochi").lower()
            
        duration_input = input("Enter travel duration in minutes (or press Enter to calculate automatically): ").strip()
        if duration_input:
            try:
                args.duration = float(duration_input)
            except ValueError:
                log_warn("Invalid duration. We will calculate it automatically using routes.")
                args.duration = None
    else:
        start_city = args.start.strip().lower() if args.start else "kumbanad"
        dest_city = args.dest.strip().lower() if args.dest else "kochi"
    
    load_dotenv()
    api_key = args.api_key or os.getenv("OPENWEATHERMAP_API_KEY", "")
    force_sim = args.simulate or not bool(api_key)
    if force_sim:
        log_warn("Simulation mode active (Offline fallback for forecasts and routing details).")

    # 1. Route Calculation & Coordinates Extraction
    route_coordinates = []
    checkpoints = []
    current_time = datetime.now()
    
    if args.route_mode == "custom":
        # Customized routing with intermediate waypoints
        waypoints = [start_city]
        if args.waypoints:
            waypoints.extend([w.strip().lower() for w in args.waypoints.split(",") if w.strip()])
        waypoints.append(dest_city)
        
        log_info(f"Custom route requested: {' -> '.join(waypoints)}")
        waypoint_coords = []
        for wp in waypoints:
            coord = geocode_location_with_fallback(wp, force_sim=force_sim)
            waypoint_coords.append(coord)
            
        legs = []
        distance_km = 0.0
        duration_mins = 0.0
        
        if args.duration is not None:
            duration_mins = args.duration
            log_info(f"Using manual travel duration override: {duration_mins} minutes")
            num_legs = len(waypoints) - 1
            for i in range(num_legs):
                dist_leg = haversine_distance(waypoint_coords[i], waypoint_coords[i+1])
                legs.append({
                    "distance_meters": dist_leg * 1000.0,
                    "duration_seconds": (duration_mins / num_legs) * 60.0
                })
                distance_km += dist_leg
        else:
            if force_sim:
                num_legs = len(waypoints) - 1
                for i in range(num_legs):
                    leg_dist_km, leg_dur_mins = estimate_travel_metrics(waypoints[i], waypoints[i+1], force_simulate=True)
                    legs.append({
                        "distance_meters": leg_dist_km * 1000.0,
                        "duration_seconds": leg_dur_mins * 60.0
                    })
                    distance_km += leg_dist_km
                    duration_mins += leg_dur_mins
            else:
                route = get_multi_driving_route(waypoint_coords)
                if route is None:
                    log_warn("Multi-waypoint routing failed. Falling back to travel simulator leg estimation.")
                    num_legs = len(waypoints) - 1
                    for i in range(num_legs):
                        leg_dist_km, leg_dur_mins = estimate_travel_metrics(waypoints[i], waypoints[i+1], force_simulate=True)
                        legs.append({
                            "distance_meters": leg_dist_km * 1000.0,
                            "duration_seconds": leg_dur_mins * 60.0
                        })
                        distance_km += leg_dist_km
                        duration_mins += leg_dur_mins
                else:
                    distance_km = round(route["distance_meters"] / 1000.0, 2)
                    duration_mins = int(round((route["duration_seconds"] / 60.0) * 1.8))
                    route_coordinates = route["coordinates"]
                    for leg in route["legs"]:
                        legs.append({
                            "distance_meters": leg["distance_meters"],
                            "duration_seconds": leg["duration_seconds"] * 1.8
                        })
                        
        # Assemble checkpoints from waypoints directly
        cumulative_duration_seconds = 0.0
        cumulative_distance_meters = 0.0
        
        for i, wp in enumerate(waypoints):
            if i == 0:
                cp_duration_mins = 0.0
                cp_dist_km = 0.0
            else:
                cumulative_duration_seconds += legs[i-1]["duration_seconds"]
                cumulative_distance_meters += legs[i-1]["distance_meters"]
                cp_duration_mins = round(cumulative_duration_seconds / 60.0, 1)
                cp_dist_km = round(cumulative_distance_meters / 1000.0, 2)
                
            cp_eta = current_time + timedelta(minutes=cp_duration_mins)
            
            index_in_route = 0
            if route_coordinates:
                min_dist = float('inf')
                for idx, coord in enumerate(route_coordinates):
                    d = haversine_distance(coord, waypoint_coords[i])
                    if d < min_dist:
                        min_dist = d
                        index_in_route = idx
                        
            checkpoints.append({
                "name": wp,
                "coords": waypoint_coords[i],
                "travel_duration": cp_duration_mins,
                "distance_from_start_km": cp_dist_km,
                "eta": cp_eta,
                "index_in_route": index_in_route
            })
            
    else:
        # Automatic routing with equidistant checkpoints
        if args.duration is not None:
            duration_mins = args.duration
            distance_km = 0.0
            log_info(f"Using manual travel duration override: {duration_mins} minutes")
        else:
            log_info(f"Estimating travel route from '{start_city}' to '{dest_city}'...")
            if force_sim:
                distance_km, duration_mins = estimate_travel_metrics(start_city, dest_city, force_simulate=True)
            else:
                lat1, lon1 = geocode_location(start_city)
                lat2, lon2 = geocode_location(dest_city)
                if lat1 is None or lat2 is None:
                    log_warn("Geocoding failed. Falling back to travel simulator.")
                    distance_km, duration_mins = estimate_travel_metrics(start_city, dest_city, force_simulate=True)
                else:
                    route = get_driving_route((lat1, lon1), (lat2, lon2))
                    if route is None:
                        log_warn("Routing service failed. Falling back to travel simulator.")
                        distance_km, duration_mins = estimate_travel_metrics(start_city, dest_city, force_simulate=True)
                    else:
                        distance_km = round(route["distance_meters"] / 1000.0, 2)
                        duration_mins = int(round((route["duration_seconds"] / 60.0) * 1.8))
                        route_coordinates = route["coordinates"]
                        
        checkpoints = get_route_checkpoints(start_city, dest_city, route_coordinates, force_simulate=force_sim)
        checkpoints = calculate_checkpoints_eta(checkpoints, route_coordinates, duration_mins, current_time)

    trip_timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    log_info(f"Journey start: {trip_timestamp}")
    log_info(f"Total checkpoints identified: {len(checkpoints)}")
    
    checkpoint_predictions = []
    
    # 4. Predict Weather & Run ML model for each Checkpoint
    for cp in checkpoints:
        cp_name = cp["name"]
        cp_coords = cp["coords"]
        cp_eta = cp["eta"]
        cp_duration = cp["travel_duration"]
        
        # A. Fetch forecast at arrival time (using Open-Meteo or simulation)
        cp_weather = fetch_checkpoint_forecast(cp_name, cp_coords, cp_eta, force_simulate=force_sim)
        
        # B. Calculate accuracy baseline
        hist_accuracy = calculate_historical_accuracy(destination_city=cp_name)
        
        # C. Assemble weather record
        record = {
            "timestamp": trip_timestamp,  # Shared timestamp to identify the trip
            "destination": cp_name,
            "travel_duration": float(cp_duration),
            "estimated_arrival_time": cp_eta.strftime("%Y-%m-%d %H:%M:%S"),
            "destination_temperature": cp_weather["destination_temperature"],
            "destination_humidity": cp_weather["destination_humidity"],
            "destination_pressure": cp_weather["destination_pressure"],
            "destination_wind_speed": cp_weather["destination_wind_speed"],
            "destination_cloud_coverage": cp_weather["destination_cloud_coverage"],
            "destination_api_rain_probability": cp_weather["destination_api_rain_probability"],
            "destination_api_prediction": cp_weather["destination_api_prediction"],
            "time_of_day": int(cp_eta.hour),
            "month": int(cp_eta.month),
            "historical_prediction_accuracy": float(hist_accuracy)
        }
        
        # D. Get ML Prediction
        model = load_or_train_checkpoint_model(cp_name)
        features_df = prepare_prediction_features(record)
        
        prediction = int(model.predict(features_df)[0])
        probabilities = model.predict_proba(features_df)[0]
        classes = list(model.classes_)
        if len(classes) == 1:
            confidence = 100.0
        else:
            try:
                class_idx = classes.index(prediction)
                confidence = float(probabilities[class_idx] * 100.0)
            except ValueError:
                confidence = 50.0

        
        # E. Log intermediate records
        log_record = record.copy()
        log_record["model_prediction"] = prediction
        log_record["model_confidence"] = confidence
        log_record["actual_rain"] = ""
        log_record["prediction_correctness"] = ""
        append_record(log_record)
        
        checkpoint_predictions.append({
            "name": cp_name,
            "eta": cp_eta,
            "weather": cp_weather,
            "prediction": prediction,
            "confidence": confidence
        })

    # 5. Generate Travel Recommendation
    recommendation = generate_route_recommendation(checkpoint_predictions)

    # 6. Display Styled Route Intelligence Report
    print("\n" + "="*70)
    print("           AI ROUTE WEATHER INTELLIGENCE REPORT")
    print("="*70)
    print(f"Route:           {start_city} ➔ {dest_city}")
    print(f"Est. Distance:   {distance_km:.2f} km")
    print(f"Total Duration:  {duration_mins} mins")
    print(f"Departure Time:  {current_time.strftime('%I:%M %p (%Y-%m-%d)')}")
    print("-"*70)
    print(f"{'CHECKPOINT':20s} | {'ETA':8s} | {'TEMP':6s} | {'HUMID':5s} | {'WIND':7s} | {'RAIN PROB':9s} | {'PREDICTION':10s}")
    print("-"*70)
    
    for pred in checkpoint_predictions:
        name = pred["name"]
        eta_str = pred["eta"].strftime("%I:%M %p")
        weather = pred["weather"]
        temp = f"{weather['destination_temperature']}°C"
        humid = f"{weather['destination_humidity']}%"
        wind = f"{weather['destination_wind_speed']}m/s"
        prob = f"{weather['destination_api_rain_probability']*100:.0f}%"
        
        # Color coding predictions for console visibility
        if pred["prediction"] == 1:
            pred_label = f"\033[91m☔ Rain ({pred['confidence']:.0f}%)\033[0m"
        else:
            pred_label = f"\033[92m☀ Clear ({pred['confidence']:.0f}%)\033[0m"
            
        print(f"{name[:20]:20s} | {eta_str:8s} | {temp:6s} | {humid:5s} | {wind:7s} | {prob:9s} | {pred_label}")
        
    print("-"*70)
    print("🔔 TRAVEL GEAR RECOMMENDATION:")
    print(f"  🚨 \033[1mRecommendation: {recommendation['gear_recommendation']}\033[0m")
    print(f"  💬 Advice: {recommendation['gear_advice']}")
    print("-"*70)
    print("🛡️ TWO-WHEELER SAFETY RIDE ADVISORY:")
    for adv in recommendation["advisories"]:
        print(f"  {adv}")
    print("="*70 + "\n")
    
    log_success("Route prediction complete and logged. Run save_feedback.py upon arrival to submit trip feedback.")

if __name__ == "__main__":
    run_prediction()
