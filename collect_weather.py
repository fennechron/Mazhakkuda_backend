"""
collect_weather.py
------------------
Responsible for gathering live weather data. 
Connects to OpenWeatherMap API (5-day forecast endpoint to get probability of precipitation)
or falls back to a realistic weather generator for mock simulation if API keys are missing.
"""

import os
import requests
import random
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from utils import log_info, log_warn, log_error

# Load environment variables from .env file if it exists
load_dotenv()

# Configuration
API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
DEFAULT_CITY = os.getenv("DEFAULT_CITY", "mannar")

def get_simulated_weather():
    """
    Generate realistic, simulated weather data to allow full pipeline testing
    without requiring an active internet connection or OpenWeatherMap API key.
    """
    log_warn("Using simulated weather data generator.")
    
    # 40% chance of a rainy setup today
    is_rainy_setup = random.random() < 0.4
    
    if is_rainy_setup:
        temperature = round(random.uniform(14.0, 22.0), 1)
        humidity = random.randint(75, 100)
        pressure = random.randint(990, 1008)
        wind_speed = round(random.uniform(5.0, 15.0), 1)
        cloud_coverage = random.randint(70, 100)
        
        # High API probability
        api_rain_probability = round(random.uniform(0.55, 0.95), 2)
        api_prediction = 1 if api_rain_probability >= 0.5 else 0
    else:
        temperature = round(random.uniform(22.0, 33.0), 1)
        humidity = random.randint(30, 65)
        pressure = random.randint(1009, 1024)
        wind_speed = round(random.uniform(1.0, 6.0), 1)
        cloud_coverage = random.randint(0, 50)
        
        # Low API probability
        api_rain_probability = round(random.uniform(0.0, 0.30), 2)
        api_prediction = 1 if api_rain_probability >= 0.5 else 0

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": wind_speed,
        "cloud_coverage": cloud_coverage,
        "api_rain_probability": api_rain_probability,
        "api_prediction": api_prediction
    }

def fetch_live_weather(api_key=None, city=None):
    """
    Fetch live weather forecast data from the OpenWeatherMap API.
    Uses the 5-day / 3-hour forecast API endpoint to retrieve the probability of precipitation (pop).
    
    Parameters:
    api_key (str): OpenWeatherMap API key.
    city (str): Name of the city.
    
    Returns:
    dict: Weather records conforming to the project features, or None on failure.
    """
    key = api_key or API_KEY
    target_city = city or DEFAULT_CITY
    
    if not key:
        log_warn("OpenWeatherMap API key is missing. Cannot fetch live data.")
        return None
        
    # We query the forecast endpoint to obtain the 'pop' (probability of precipitation) parameter
    url = f"https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": target_city,
        "appid": key,
        "units": "metric",
        "cnt": 1  # We only need the immediate upcoming 3-hour forecast block
    }
    
    try:
        log_info(f"Fetching live weather forecast for '{target_city}' from OpenWeatherMap...")
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            log_error(f"API Request failed with status code {response.status_code}: {response.text}")
            return None
            
        data = response.json()
        
        if "list" not in data or len(data["list"]) == 0:
            log_error("Unexpected API response structure. No forecast records found.")
            return None
            
        forecast = data["list"][0]
        
        # Extract features
        temperature = forecast["main"]["temp"]
        humidity = forecast["main"]["humidity"]
        pressure = forecast["main"]["pressure"]
        wind_speed = forecast["wind"]["speed"]
        cloud_coverage = forecast["clouds"]["all"]
        
        # OWM API represents 'pop' (probability of precipitation) as float between 0.0 and 1.0
        api_rain_probability = forecast.get("pop", 0.0)
        api_prediction = 1 if api_rain_probability >= 0.5 else 0
        
        weather_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": float(temperature),
            "humidity": int(humidity),
            "pressure": int(pressure),
            "wind_speed": float(wind_speed),
            "cloud_coverage": int(cloud_coverage),
            "api_rain_probability": float(api_rain_probability),
            "api_prediction": int(api_prediction)
        }
        
        log_info(f"Successfully collected live weather for '{target_city}': Temp={temperature}°C, Humidity={humidity}%, API Rain Prob={api_rain_probability*100}%")
        return weather_data
        
    except requests.exceptions.RequestException as e:
        log_error(f"Network error while connecting to OpenWeatherMap API: {e}")
        return None
    except KeyError as e:
        log_error(f"Parser error extracting data from API payload: missing key {e}")
        return None
    except Exception as e:
        log_error(f"Unexpected error in live weather collection: {e}")
        return None

def collect_weather_record(api_key=None, city=None, force_simulate=False):
    """
    Unified entry point. Attempts to collect live data first. 
    If it fails, lacks keys, or if force_simulate is True, returns simulated weather data.
    """
    if force_simulate:
        return get_simulated_weather()
        
    live_data = fetch_live_weather(api_key, city)
    if live_data is not None:
        return live_data
        
    log_warn("Falling back to simulated weather generation...")
    return get_simulated_weather()

if __name__ == "__main__":
    # Test script standalone
    print("--- Running standalone weather collection test ---")
    data = collect_weather_record(force_simulate=False)
    print("Collected Data Block:")
    for k, v in data.items():
        print(f"  {k}: {v}")
