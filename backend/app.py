import os
import sys
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import pandas as pd

# Add the parent directory to sys.path so we can import its modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import (
    load_model, append_record, calculate_historical_accuracy, load_dataset, update_trip_feedback
)
from travel_time_estimator import estimate_travel_metrics
from route_service import geocode_location, get_driving_route, get_multi_driving_route
from preprocess import prepare_prediction_features
from train_model import train_model
from route_checkpoints import get_route_checkpoints
from eta_calculator import calculate_checkpoints_eta, haversine_distance
from route_weather_forecast import fetch_checkpoint_forecast
from travel_recommendation import generate_route_recommendation
from predict_model import load_or_train_checkpoint_model, geocode_location_with_fallback

load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route('/api/predict/travel', methods=['POST'])
def predict_travel():
    data = request.json or {}
    
    start_city = data.get('start', 'kumbanad').strip().lower()
    dest_city = data.get('dest', 'kochi').strip().lower()
    duration = data.get('duration') # optional manual duration in mins
    route_mode = data.get('route_mode', 'auto')
    waypoints_str = data.get('waypoints', '')
    force_sim = data.get('simulate', False)
    api_key = data.get('api_key', os.getenv("OPENWEATHERMAP_API_KEY", ""))
    
    if not api_key:
        force_sim = True

    route_coordinates = []
    checkpoints = []
    current_time = datetime.now()
    
    legs = []
    distance_km = 0.0
    duration_mins = 0.0
    
    if route_mode == "custom":
        waypoints = [start_city]
        if waypoints_str:
            waypoints.extend([w.strip().lower() for w in waypoints_str.split(",") if w.strip()])
        waypoints.append(dest_city)
        
        waypoint_coords = []
        for wp in waypoints:
            coord = geocode_location_with_fallback(wp, force_sim=force_sim)
            waypoint_coords.append(coord)
            
        if duration is not None:
            duration_mins = float(duration)
            num_legs = len(waypoints) - 1
            for i in range(num_legs):
                dist_leg = haversine_distance(waypoint_coords[i], waypoint_coords[i+1])
                legs.append({
                    "distance_meters": dist_leg * 1000.0,
                    "duration_seconds": (duration_mins / num_legs) * 60.0
                })
                distance_km += dist_leg
        else:
            route = get_multi_driving_route(waypoint_coords)
            if route is None:
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
        if duration is not None:
            duration_mins = float(duration)
            distance_km = 0.0
        else:
            lat1, lon1 = geocode_location(start_city)
            lat2, lon2 = geocode_location(dest_city)
            if lat1 is None or lat2 is None:
                distance_km, duration_mins = estimate_travel_metrics(start_city, dest_city, force_simulate=True)
            else:
                route = get_driving_route((lat1, lon1), (lat2, lon2))
                if route is None:
                    distance_km, duration_mins = estimate_travel_metrics(start_city, dest_city, force_simulate=True)
                else:
                    distance_km = round(route["distance_meters"] / 1000.0, 2)
                    duration_mins = int(round((route["duration_seconds"] / 60.0) * 1.8))
                    route_coordinates = route["coordinates"]
                        
        checkpoints = get_route_checkpoints(start_city, dest_city, route_coordinates, force_simulate=force_sim)
        checkpoints = calculate_checkpoints_eta(checkpoints, route_coordinates, duration_mins, current_time)

    trip_timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    checkpoint_predictions = []
    
    for cp in checkpoints:
        cp_name = cp["name"]
        cp_coords = cp["coords"]
        cp_eta = cp["eta"]
        cp_duration = cp["travel_duration"]
        
        cp_weather = fetch_checkpoint_forecast(cp_name, cp_coords, cp_eta, force_simulate=force_sim)
        hist_accuracy = calculate_historical_accuracy(destination_city=cp_name)
        
        record = {
            "timestamp": trip_timestamp,
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

        log_record = record.copy()
        log_record["model_prediction"] = prediction
        log_record["model_confidence"] = confidence
        log_record["actual_rain"] = ""
        log_record["prediction_correctness"] = ""
        append_record(log_record)
        
        checkpoint_predictions.append({
            "name": cp_name,
            "coords": cp_coords,
            "distance_from_start_km": cp.get("distance_from_start_km", 0.0),
            "eta": cp_eta.strftime("%Y-%m-%d %H:%M:%S"),
            "weather": cp_weather,
            "prediction": prediction,
            "confidence": confidence
        })

    recommendation = generate_route_recommendation(checkpoint_predictions)

    return jsonify({
        "success": True,
        "trip_timestamp": trip_timestamp,
        "route": f"{start_city} to {dest_city}",
        "distance_km": distance_km,
        "duration_mins": duration_mins,
        "checkpoints": checkpoint_predictions,
        "recommendation": recommendation
    })

@app.route('/api/feedback/travel', methods=['POST'])
def travel_feedback():
    data = request.json or {}
    timestamp = data.get('timestamp')
    feedbacks = data.get('feedbacks') # List of dicts with {'destination': 'city', 'actual_rain': 1/0}

    if not timestamp or not feedbacks:
        return jsonify({"success": False, "message": "timestamp and feedbacks array are required"}), 400

    feedback_map = { f['destination'].lower(): f['actual_rain'] for f in feedbacks }
    
    print(f"Feedback Map: {feedback_map}")
    
    updated_count = update_trip_feedback(timestamp, feedback_map)

    if updated_count > 0:
        return jsonify({"success": True, "message": f"Updated feedback for {updated_count} checkpoints."})
    else:
        return jsonify({"success": False, "message": "No matching checkpoints found to update."}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
