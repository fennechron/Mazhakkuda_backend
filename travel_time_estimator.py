"""
travel_time_estimator.py
------------------------
Responsible for estimating travel durations and distances.
Utilizes route_service.py to fetch real road network estimates
and implements a robust simulator fallback for offline testing or missing network setups.
"""

import random
from route_service import geocode_location, get_driving_route
from utils import log_info, log_warn, log_error

# Predefined coordinates and distances for common local routing simulations (Kerala, India)
MOCK_DATABASE = {
    ("kumbanad", "chengannur"): {"distance_km": 11.2, "duration_mins": 20},
    ("chengannur", "kumbanad"): {"distance_km": 11.2, "duration_mins": 20},
    ("thiruvalla", "chengannur"): {"distance_km": 10.1, "duration_mins": 18},
    ("chengannur", "thiruvalla"): {"distance_km": 10.1, "duration_mins": 18},
    ("kozhencherry", "chengannur"): {"distance_km": 13.5, "duration_mins": 25},
    ("chengannur", "kozhencherry"): {"distance_km": 13.5, "duration_mins": 25},
    ("mannar", "chengannur"): {"distance_km": 9.8, "duration_mins": 16},
    ("chengannur", "mannar"): {"distance_km": 9.8, "duration_mins": 16},
    ("kumbanad", "thiruvalla"): {"distance_km": 8.5, "duration_mins": 15},
    ("thiruvalla", "kumbanad"): {"distance_km": 8.5, "duration_mins": 15}
}

def get_simulated_travel_metrics(start_loc, dest_loc):
    """
    Generate realistic travel metrics for simulation.
    If the start/destination matches our mock database, return exact estimates.
    Otherwise, generate randomized driving statistics.
    """
    start_key = start_loc.lower().strip()
    dest_key = dest_loc.lower().strip()
    
    match = MOCK_DATABASE.get((start_key, dest_key))
    if match:
        log_info(f"[SIMULATOR] Match found for routing: {start_loc} -> {dest_loc}")
        return match["distance_km"], match["duration_mins"]
        
    # Generate randomized route metrics: 5 to 50 km at 40 km/h average speed
    distance = round(random.uniform(5.0, 50.0), 1)
    # 40 km/h = 1.5 mins per km, plus some congestion buffer
    duration = int(distance * random.uniform(1.2, 1.8))
    log_info(f"[SIMULATOR] Simulated route generated: {distance} km, {duration} mins")
    return distance, duration

def estimate_travel_metrics(start_location, destination_location, force_simulate=False):
    """
    Estimates travel distance (km) and driving duration (minutes) between two locations.
    Attempts to retrieve real coordinates and route from OSRM unless force_simulate is set.
    
    Parameters:
    start_location (str): Starting location name.
    destination_location (str): Destination location name.
    force_simulate (bool): If True, bypass API calls and use the simulator.
    
    Returns:
    tuple: (distance_km, duration_minutes)
    """
    if force_simulate:
        return get_simulated_travel_metrics(start_location, destination_location)
        
    # Attempt Geocoding
    lat1, lon1 = geocode_location(start_location)
    lat2, lon2 = geocode_location(destination_location)
    
    if lat1 is None or lat2 is None:
        log_warn("Geocoding failed. Falling back to simulated travel estimation.")
        return get_simulated_travel_metrics(start_location, destination_location)
        
    # Attempt Route Calculation
    route = get_driving_route((lat1, lon1), (lat2, lon2))
    if route is None:
        log_warn("Route service failed. Falling back to simulated travel estimation.")
        return get_simulated_travel_metrics(start_location, destination_location)
        
    distance_km = round(route["distance_meters"] / 1000.0, 2)
    # Apply a local traffic/two-wheeler scaling factor of 1.8x to the raw OSRM driving duration
    duration_minutes = int(round((route["duration_seconds"] / 60.0) * 1.8))
    
    return distance_km, duration_minutes

if __name__ == "__main__":
    print("--- Travel Time Estimator stand-alone test ---")
    start = "Kumbanad"
    dest = "Chengannur"
    
    dist, dur = estimate_travel_metrics(start, dest, force_simulate=False)
    print(f"Results for '{start}' to '{dest}':")
    print(f"  Distance: {dist} km")
    print(f"  Duration: {dur} minutes")
