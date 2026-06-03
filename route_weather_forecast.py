"""
route_weather_forecast.py
-------------------------
Fetches hourly weather forecasts for a route checkpoint coordinate at its estimated arrival time (ETA)
using the free, public Open-Meteo API. Falls back to a mock weather simulator if offline.
"""

import requests
import random
from datetime import datetime
from utils import log_info, log_warn, log_error

def get_simulated_checkpoint_forecast(location_name, eta):
    """
    Generate realistic simulated weather forecast data for a checkpoint.
    """
    log_warn(f"[SIMULATOR] Generating simulated forecast for checkpoint '{location_name}' at {eta.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 35% chance of rain setup
    is_rainy = random.random() < 0.35
    
    if is_rainy:
        temp = round(random.uniform(18.0, 24.0), 1)
        humidity = random.randint(75, 100)
        pressure = random.randint(990, 1008)
        wind_speed = round(random.uniform(3.5, 9.5), 1)
        cloud_coverage = random.randint(70, 100)
        api_prob = round(random.uniform(0.55, 0.95), 2)
        api_pred = 1
    else:
        temp = round(random.uniform(25.0, 32.0), 1)
        humidity = random.randint(40, 68)
        pressure = random.randint(1009, 1020)
        wind_speed = round(random.uniform(1.0, 4.5), 1)
        cloud_coverage = random.randint(0, 40)
        api_prob = round(random.uniform(0.0, 0.25), 2)
        api_pred = 0
        
    return {
        "destination_temperature": temp,
        "destination_humidity": humidity,
        "destination_pressure": pressure,
        "destination_wind_speed": wind_speed,
        "destination_cloud_coverage": cloud_coverage,
        "destination_api_rain_probability": api_prob,
        "destination_api_prediction": api_pred
    }

def fetch_checkpoint_forecast(location_name, coords, eta, force_simulate=False):
    """
    Queries Open-Meteo hourly forecast API for the closest forecast slot to the ETA.
    
    Parameters:
    location_name (str): Name of the checkpoint.
    coords (tuple): (latitude, longitude) coordinates.
    eta (datetime): Estimated arrival time.
    force_simulate (bool): If True, forces simulation fallback.
    
    Returns:
    dict: Weather forecast data structured for prediction.
    """
    if force_simulate or not coords or coords == (0.0, 0.0):
        return get_simulated_checkpoint_forecast(location_name, eta)
        
    lat, lon = coords
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,cloud_cover,precipitation_probability",
        "wind_speed_unit": "ms", # meters per second to align with OpenWeatherMap
        "forecast_days": 3
    }
    
    try:
        log_info(f"Fetching Open-Meteo hourly forecast for '{location_name}' at ({lat:.4f}, {lon:.4f})...")
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            log_error(f"Open-Meteo API failed with code {response.status_code}: {response.text}")
            return get_simulated_checkpoint_forecast(location_name, eta)
            
        data = response.json()
        if "hourly" not in data or "time" not in data["hourly"]:
            log_error("Invalid hourly forecast payload from Open-Meteo.")
            return get_simulated_checkpoint_forecast(location_name, eta)
            
        # Parse ISO timestamps into datetime objects
        times = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
        
        # Find closest slot to ETA
        diffs = [abs((t - eta).total_seconds()) for t in times]
        closest_idx = diffs.index(min(diffs))
        
        hourly_data = data["hourly"]
        
        # Extract features at closest index
        temp = hourly_data["temperature_2m"][closest_idx]
        humidity = hourly_data["relative_humidity_2m"][closest_idx]
        pressure = hourly_data["surface_pressure"][closest_idx]
        wind_speed = hourly_data["wind_speed_10m"][closest_idx]
        cloud_coverage = hourly_data["cloud_cover"][closest_idx]
        
        # Open-Meteo probability is 0-100, our system uses 0.0-1.0
        api_prob = float(hourly_data["precipitation_probability"][closest_idx]) / 100.0
        api_pred = 1 if api_prob >= 0.5 else 0
        
        matched_time_str = times[closest_idx].strftime("%Y-%m-%d %H:%M")
        log_info(f"Matched Open-Meteo hourly slot at {matched_time_str} (ETA: {eta.strftime('%Y-%m-%d %H:%M')})")
        log_info(f"Forecast for {location_name}: Temp={temp}°C, Humidity={humidity}%, Wind={wind_speed}m/s, Rain Prob={api_prob * 100:.0f}%")
        
        return {
            "destination_temperature": float(temp),
            "destination_humidity": int(humidity),
            "destination_pressure": int(pressure) if pressure is not None else 1013,
            "destination_wind_speed": float(wind_speed),
            "destination_cloud_coverage": int(cloud_coverage),
            "destination_api_rain_probability": api_prob,
            "destination_api_prediction": api_pred
        }
        
    except Exception as e:
        log_error(f"Error fetching/parsing Open-Meteo forecast: {e}")
        return get_simulated_checkpoint_forecast(location_name, eta)

if __name__ == "__main__":
    print("--- Test Open-Meteo Forecast ---")
    eta = datetime.now()
    coords = (9.3920, 76.6341)
    res = fetch_checkpoint_forecast("Kumbanad", coords, eta, force_simulate=False)
    print("Result:", res)
