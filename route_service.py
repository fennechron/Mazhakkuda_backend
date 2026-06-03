"""
route_service.py
----------------
Provides routing calculations and geocoding services.
Queries OpenStreetMap's Nominatim API to convert city names to coordinates (lat/lon)
and queries the Open Source Routing Machine (OSRM) API to calculate estimated driving time and distance.
"""

import requests
from utils import log_info, log_warn, log_error

# Custom User-Agent to comply with Nominatim's usage policy
USER_AGENT = "TravelWeatherPredictor/1.0 (contact: travel-weather-predictor@fennechron.org)"
HEADERS = {"User-Agent": USER_AGENT}

def geocode_location(location_name):
    """
    Convert a location name (e.g., 'Chengannur') into latitude and longitude coordinates
    using the OpenStreetMap Nominatim Geocoding API.
    
    Parameters:
    location_name (str): Name of the town/city.
    
    Returns:
    tuple: (latitude, longitude) as floats, or (None, None) on failure.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": location_name,
        "format": "json",
        "limit": 1
    }
    
    try:
        log_info(f"Geocoding location '{location_name}' using Nominatim...")
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        
        if response.status_code != 200:
            log_error(f"Geocoding request failed with status code {response.status_code}: {response.text}")
            return None, None
            
        data = response.json()
        if not data or len(data) == 0:
            log_warn(f"No coordinates found for location: '{location_name}'")
            return None, None
            
        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
        log_info(f"Geocoded '{location_name}' to ({lat:.5f}, {lon:.5f})")
        return lat, lon
        
    except requests.exceptions.RequestException as e:
        log_error(f"Network error during geocoding: {e}")
        return None, None
    except (KeyError, ValueError) as e:
        log_error(f"Parsing error during geocoding response: {e}")
        return None, None
    except Exception as e:
        log_error(f"Unexpected error in geocode_location: {e}")
        return None, None

def get_multi_driving_route(coordinates_list):
    """
    Retrieve driving route traversing a list of coordinate points in order.
    
    Parameters:
    coordinates_list (list of tuples): List of (latitude, longitude) coordinates.
    
    Returns:
    dict: Dictionary containing:
          - 'distance_meters': Total distance.
          - 'duration_seconds': Total duration.
          - 'coordinates': Full route coordinates list.
          - 'legs': List of dictionaries with leg information (distance and duration).
          Or None on failure.
    """
    if len(coordinates_list) < 2:
        log_error("get_multi_driving_route requires at least two coordinate points.")
        return None
        
    # OSRM coordinates are specified in {longitude},{latitude};{longitude},{latitude} format
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in coordinates_list])
    url = f"http://router.project-osrm.org/route/v1/driving/{coords_str}"
    params = {
        "overview": "full",
        "geometries": "geojson"
    }
    
    try:
        log_info(f"Calculating multi-waypoint driving route for {len(coordinates_list)} points...")
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        
        if response.status_code != 200:
            log_error(f"OSRM Routing request failed with status code {response.status_code}: {response.text}")
            return None
            
        data = response.json()
        if "routes" not in data or len(data["routes"]) == 0:
            log_warn("No routes found between the provided coordinates.")
            return None
            
        route = data["routes"][0]
        distance = float(route["distance"])   # in meters
        duration = float(route["duration"])   # in seconds
        geojson_coords = route.get("geometry", {}).get("coordinates", [])
        # Convert [lon, lat] to (lat, lon)
        coordinates = [(c[1], c[0]) for c in geojson_coords]
        
        # Parse legs
        legs_data = []
        for leg in route.get("legs", []):
            legs_data.append({
                "distance_meters": float(leg.get("distance", 0.0)),
                "duration_seconds": float(leg.get("duration", 0.0))
            })
            
        log_info(f"Multi-waypoint route found: {distance / 1000.0:.2f} km, Duration: {duration / 60.0:.1f} mins, Legs: {len(legs_data)}")
        return {
            "distance_meters": distance,
            "duration_seconds": duration,
            "coordinates": coordinates,
            "legs": legs_data
        }
        
    except requests.exceptions.RequestException as e:
        log_error(f"Network error during multi-waypoint route calculation: {e}")
        return None
    except (KeyError, ValueError) as e:
        log_error(f"Parsing error during multi-waypoint routing response: {e}")
        return None
    except Exception as e:
        log_error(f"Unexpected error in get_multi_driving_route: {e}")
        return None

def get_driving_route(start_coords, dest_coords):
    """
    Retrieve estimated driving distance and travel duration between two coordinates
    using the Open Source Routing Machine (OSRM) public API.
    
    Parameters:
    start_coords (tuple): (latitude, longitude) of the start point.
    dest_coords (tuple): (latitude, longitude) of the destination.
    
    Returns:
    dict: Dictionary containing 'distance_meters' and 'duration_seconds', or None on failure.
    """
    return get_multi_driving_route([start_coords, dest_coords])

if __name__ == "__main__":
    print("--- Route Service Standalone Test ---")
    start = "Kumbanad"
    dest = "Chengannur"
    
    lat1, lon1 = geocode_location(start)
    lat2, lon2 = geocode_location(dest)
    
    if lat1 and lat2:
        route = get_driving_route((lat1, lon1), (lat2, lon2))
        print("Route result:", route)
    else:
        print("Geocoding failed.")
