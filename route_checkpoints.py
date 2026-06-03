"""
route_checkpoints.py
--------------------
Handles extracting a list of checkpoints along a travel route.
Combines Nominatim reverse geocoding with a cached memory store,
and provides a robust offline fallback database of common routes in Kerala.
"""

import time
import requests
import json
import os
from utils import log_info, log_warn, log_error

USER_AGENT = "TravelWeatherPredictor/1.0 (contact: travel-weather-predictor@fennechron.org)"
HEADERS = {"User-Agent": USER_AGENT}

# Local JSON cache for geocoded location names to avoid hitting the API
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "nominatim_cache.json")

# Predefined route checkpoints fallback
PREDEFINED_ROUTES = {
    ("kumbanad", "kochi"): [
        {"name": "Kumbanad", "coords": (9.3920, 76.6341)},
        {"name": "Chengannur", "coords": (9.3175, 76.6111)},
        {"name": "Mavelikara", "coords": (9.2435, 76.5492)},
        {"name": "Haripad", "coords": (9.2847, 76.4357)},
        {"name": "Alappuzha", "coords": (9.4981, 76.3388)},
        {"name": "Kochi", "coords": (9.9312, 76.2673)}
    ],
    ("kumbanad", "chengannur"): [
        {"name": "Kumbanad", "coords": (9.3920, 76.6341)},
        {"name": "Kallissery", "coords": (9.3400, 76.6150)},
        {"name": "Chengannur", "coords": (9.3175, 76.6111)}
    ],
    ("kumbanad", "thiruvalla"): [
        {"name": "Kumbanad", "coords": (9.3920, 76.6341)},
        {"name": "Muthoor", "coords": (9.3950, 76.5900)},
        {"name": "Thiruvalla", "coords": (9.3828, 76.5724)}
    ],
    ("chengannur", "kochi"): [
        {"name": "Chengannur", "coords": (9.3175, 76.6111)},
        {"name": "Mavelikara", "coords": (9.2435, 76.5492)},
        {"name": "Haripad", "coords": (9.2847, 76.4357)},
        {"name": "Alappuzha", "coords": (9.4981, 76.3388)},
        {"name": "Kochi", "coords": (9.9312, 76.2673)}
    ],
    ("chengannur", "trivandrum"): [
        {"name": "Chengannur", "coords": (9.3175, 76.6111)},
        {"name": "Pandalam", "coords": (9.2272, 76.6853)},
        {"name": "Adoor", "coords": (9.1534, 76.7329)},
        {"name": "Kottarakkara", "coords": (8.9982, 76.7797)},
        {"name": "Trivandrum", "coords": (8.5241, 76.9366)}
    ]
}

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache):
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=4)
    except Exception as e:
        log_warn(f"Failed to save Nominatim cache: {e}")

_cache = load_cache()

def reverse_geocode(lat, lon):
    """
    Attempt to find the place name for a set of coordinates using OSM Nominatim.
    Uses memory/disk cache first.
    """
    key = f"{lat:.4f},{lon:.4f}"
    if key in _cache:
        return _cache[key]
        
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "accept-language": "en"
    }
    
    try:
        log_info(f"Reverse geocoding coordinate ({lat:.4f}, {lon:.4f})...")
        # Throttle to comply with Nominatim's 1 req/sec policy
        time.sleep(1.0)
        response = requests.get(url, params=params, headers=HEADERS, timeout=5)
        if response.status_code == 200:
            data = response.json()
            address = data.get("address", {})
            
            # Find the best name
            name = (
                address.get("village") or 
                address.get("town") or 
                address.get("city") or 
                address.get("suburb") or 
                address.get("hamlet") or 
                address.get("municipality") or 
                address.get("county") or
                data.get("name") or
                f"Point ({lat:.3f}, {lon:.3f})"
            )
            name = str(name).strip()
            _cache[key] = name
            save_cache(_cache)
            return name
        else:
            log_warn(f"Nominatim reverse geocode failed with code {response.status_code}")
            return f"Point ({lat:.3f}, {lon:.3f})"
    except Exception as e:
        log_error(f"Error during reverse geocoding: {e}")
        return f"Point ({lat:.3f}, {lon:.3f})"

def get_route_checkpoints(start_name, dest_name, route_coordinates, force_simulate=False):
    """
    Identifies 5-6 checkpoints along the route.
    If simulation is forced or geocoding fails, returns fallback values.
    """
    start_key = start_name.lower().strip()
    dest_key = dest_name.lower().strip()
    
    checkpoints = []
    
    # Check predefined DB first (especially helpful for simulate mode or offline)
    if force_simulate or not route_coordinates:
        if (start_key, dest_key) in PREDEFINED_ROUTES:
            log_info(f"Using predefined checkpoints for route: {start_name} -> {dest_name}")
            checkpoints = PREDEFINED_ROUTES[(start_key, dest_key)]
            
    # Sample checkpoints from actual route coordinates
    if not checkpoints and route_coordinates and len(route_coordinates) >= 5:
        n = len(route_coordinates)
        # Sample 5 points (start, 25%, 50%, 75%, destination)
        sample_indices = [0, int(n * 0.25), int(n * 0.50), int(n * 0.75), n - 1]
        
        for i, idx in enumerate(sample_indices):
            lat, lon = route_coordinates[idx]
            if i == 0:
                name = start_name
            elif i == len(sample_indices) - 1:
                name = dest_name
            else:
                # If simulate is forced, we construct a mock name to avoid API calls
                if force_simulate:
                    name = f"Route Point {i} ({lat:.2f}, {lon:.2f})"
                else:
                    name = reverse_geocode(lat, lon)
            
            checkpoints.append({
                "name": name,
                "coords": (lat, lon),
                "index_in_route": idx
            })
            
    # Final absolute fallback if no route coordinates are available
    if not checkpoints:
        log_warn("No route coordinates available. Falling back to simple start/destination list.")
        checkpoints = [
            {"name": start_name, "coords": (0.0, 0.0), "index_in_route": 0},
            {"name": f"Midpoint on route", "coords": (0.0, 0.0), "index_in_route": 1},
            {"name": dest_name, "coords": (0.0, 0.0), "index_in_route": 2}
        ]

    # Convert all checkpoint names to lower case and deduplicate
    final_checkpoints = []
    seen_names = set()
    for cp in checkpoints:
        name = cp["name"].lower().strip()
        if name not in seen_names:
            final_checkpoints.append({
                "name": name,
                "coords": cp["coords"],
                "index_in_route": cp.get("index_in_route", 0)
            })
            seen_names.add(name)
    return final_checkpoints

if __name__ == "__main__":
    print("--- Test Route Checkpoints ---")
    points = get_route_checkpoints("Kumbanad", "Kochi", [], force_simulate=True)
    print("Checkpoints:", points)
