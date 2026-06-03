"""
eta_calculator.py
-----------------
Calculates the ETA and cumulative travel duration for each checkpoint along a travel route,
proportionate to its actual road distance from the starting location.
"""

import math
from datetime import datetime, timedelta

def haversine_distance(coord1, coord2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

def calculate_checkpoints_eta(checkpoints, route_coordinates, total_duration_mins, start_time=None):
    """
    Computes ETAs for a list of checkpoints based on distance along the route.
    
    Parameters:
    checkpoints (list): List of dictionaries, each representing a checkpoint.
    route_coordinates (list): List of (lat, lon) coordinates of the route.
    total_duration_mins (float): Total travel duration in minutes.
    start_time (datetime): Departure time.
    
    Returns:
    list: Updated checkpoints list with 'travel_duration' and 'eta' keys.
    """
    if start_time is None:
        start_time = datetime.now()
        
    n = len(checkpoints)
    
    # If no route coordinates, or coordinates are insufficient, fall back to linear time interpolation and straight-line distance
    if not route_coordinates or len(route_coordinates) < 2:
        total_sim_dist = 0.0
        cp_distances = [0.0]
        for i in range(1, len(checkpoints)):
            coord1 = checkpoints[i-1].get("coords")
            coord2 = checkpoints[i].get("coords")
            if coord1 and coord2 and coord1 != (0.0, 0.0) and coord2 != (0.0, 0.0):
                dist = haversine_distance(coord1, coord2)
            else:
                dist = 15.0  # default mock distance in km per checkpoint
            total_sim_dist += dist
            cp_distances.append(total_sim_dist)
            
        for i, cp in enumerate(checkpoints):
            prop = float(i) / (n - 1) if n > 1 else 0.0
            duration = prop * total_duration_mins
            cp["travel_duration"] = round(duration, 1)
            cp["eta"] = start_time + timedelta(minutes=duration)
            cp["distance_from_start_km"] = round(cp_distances[i], 2)
        return checkpoints
        
    # Calculate cumulative distance along route coordinates
    cumulative_dist = [0.0]
    total_dist = 0.0
    for i in range(1, len(route_coordinates)):
        dist = haversine_distance(route_coordinates[i-1], route_coordinates[i])
        total_dist += dist
        cumulative_dist.append(total_dist)
        
    # Assign durations and ETAs
    for cp in checkpoints:
        idx = cp.get("index_in_route", 0)
        idx = max(0, min(idx, len(route_coordinates) - 1))
        
        cp_dist = cumulative_dist[idx]
        
        # Calculate proportion of distance covered
        prop = cp_dist / total_dist if total_dist > 0.0 else 0.0
        duration = prop * total_duration_mins
        
        cp["travel_duration"] = round(duration, 1)
        cp["eta"] = start_time + timedelta(minutes=duration)
        cp["distance_from_start_km"] = round(cp_dist, 2)
        
    return checkpoints

if __name__ == "__main__":
    print("--- Test ETA Calculator ---")
    cps = [
        {"name": "Start", "index_in_route": 0},
        {"name": "Midpoint", "index_in_route": 2},
        {"name": "End", "index_in_route": 4}
    ]
    coords = [
        (9.3920, 76.6341),
        (9.3500, 76.6200),
        (9.3175, 76.6111),
        (9.2800, 76.5800),
        (9.2435, 76.5492)
    ]
    res = calculate_checkpoints_eta(cps, coords, 40)
    for c in res:
        print(f"{c['name']} -> Travel Duration: {c['travel_duration']} mins, ETA: {c['eta']}, Dist: {c['distance_from_start_km']} km")
