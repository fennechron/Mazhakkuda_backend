"""
destination_forecast.py
------------------------
Responsible for gathering weather forecast parameters at the destination city
matching the user's estimated arrival time.
Queries OpenWeatherMap 5-day/3-hour forecast API, identifies the forecast slot
closest to the arrival time, and extracts features.
Provides a mock generator if API queries fail.
"""

import os
import requests
import random
from datetime import datetime
from dotenv import load_dotenv
from utils import log_info, log_warn, log_error

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")

def get_simulated_destination_forecast(city, arrival_time_str):
    """
    Generate realistic simulated weather forecast data for the destination city at arrival time.
    """
    log_warn(f"[SIMULATOR] Generating simulated forecast for '{city}' at {arrival_time_str}")
    
    # Randomly select a weather state (35% chance of rain)
    is_rainy = random.random() < 0.35
    
    if is_rainy:
        temp = round(random.uniform(18.0, 25.0), 1)
        humidity = random.randint(75, 100)
        pressure = random.randint(992, 1008)
        wind_speed = round(random.uniform(4.0, 11.0), 1)
        cloud_coverage = random.randint(70, 100)
        api_prob = round(random.uniform(0.55, 0.95), 2)
        api_pred = 1
    else:
        temp = round(random.uniform(24.0, 32.0), 1)
        humidity = random.randint(35, 65)
        pressure = random.randint(1009, 1022)
        wind_speed = round(random.uniform(1.0, 5.0), 1)
        cloud_coverage = random.randint(0, 45)
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

def fetch_destination_forecast(city, arrival_time_str, api_key=None, force_simulate=False):
    """
    Fetches destination weather forecast at the estimated time of arrival.
    
    Parameters:
    city (str): Destination city name.
    arrival_time_str (str): Estimated arrival time in 'YYYY-MM-DD HH:MM:SS' format.
    api_key (str): OpenWeatherMap API Key.
    force_simulate (bool): If True, forces simulation fallback.
    
    Returns:
    dict: Dictionary containing forecasted weather features.
    """
    key = api_key or API_KEY
    
    if force_simulate or not key:
        return get_simulated_destination_forecast(city, arrival_time_str)
        
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": city,
        "appid": key,
        "units": "metric"
    }
    
    try:
        log_info(f"Fetching OWM 5-day forecast for '{city}' to align with arrival time...")
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            log_error(f"OWM API request failed (Status {response.status_code}): {response.text}")
            return get_simulated_destination_forecast(city, arrival_time_str)
            
        data = response.json()
        if "list" not in data or len(data["list"]) == 0:
            log_error("Invalid forecast payload structure.")
            return get_simulated_destination_forecast(city, arrival_time_str)
            
        # Parse target arrival time
        target_dt = datetime.strptime(arrival_time_str, "%Y-%m-%d %H:%M:%S").timestamp()
        
        # Find the forecast slot closest to the arrival time
        closest_slot = None
        min_time_diff = float("inf")
        
        for slot in data["list"]:
            slot_dt = slot["dt"]
            diff = abs(slot_dt - target_dt)
            if diff < min_time_diff:
                min_time_diff = diff
                closest_slot = slot
                
        if closest_slot is None:
            log_error("Could not find a matching forecast slot.")
            return get_simulated_destination_forecast(city, arrival_time_str)
            
        # Extract forecast parameters
        temp = closest_slot["main"]["temp"]
        humidity = closest_slot["main"]["humidity"]
        pressure = closest_slot["main"]["pressure"]
        wind_speed = closest_slot["wind"]["speed"]
        cloud_coverage = closest_slot["clouds"]["all"]
        
        # OWM API represents 'pop' (probability of precipitation) as float between 0.0 and 1.0
        api_rain_probability = closest_slot.get("pop", 0.0)
        api_prediction = 1 if api_rain_probability >= 0.5 else 0
        
        slot_time_str = datetime.fromtimestamp(closest_slot["dt"]).strftime("%Y-%m-%d %H:%M:%S")
        log_info(f"Matched OWM forecast slot at {slot_time_str} (Arrival: {arrival_time_str})")
        log_info(f"Forecasted Temp={temp}°C, Humidity={humidity}%, OWM Rain Prob={api_rain_probability*100}%")
        
        return {
            "destination_temperature": float(temp),
            "destination_humidity": int(humidity),
            "destination_pressure": int(pressure),
            "destination_wind_speed": float(wind_speed),
            "destination_cloud_coverage": int(cloud_coverage),
            "destination_api_rain_probability": float(api_rain_probability),
            "destination_api_prediction": int(api_prediction)
        }
        
    except Exception as e:
        log_error(f"Exception encountered during forecast matching: {e}")
        return get_simulated_destination_forecast(city, arrival_time_str)

if __name__ == "__main__":
    print("--- Destination Forecast Standalone Test ---")
    city = "Chengannur"
    from datetime import datetime, timedelta
    arrival = (datetime.now() + timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M:%S")
    
    forecast = fetch_destination_forecast(city, arrival, force_simulate=False)
    print("Matched Forecast:", forecast)
