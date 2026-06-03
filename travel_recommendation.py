"""
travel_recommendation.py
-------------------------
Analyzes route-wide weather predictions and produces gear recommendations
(umbrella, raincoat, or no protection) and safety advisories for travelers, particularly two-wheeler riders.
"""

def generate_route_recommendation(predictions):
    """
    Analyzes checkpoint predictions to provide gear recommendations and riding advisories.
    
    Parameters:
    predictions (list): List of dictionaries, each containing:
                        - 'name': checkpoint name
                        - 'eta': arrival time datetime
                        - 'weather': weather forecast parameters
                        - 'prediction': ML prediction (1 = Rain, 0 = No Rain)
                        - 'confidence': ML prediction confidence percentage
                        
    Returns:
    dict: Recommendations and ride advisories.
    """
    rain_checkpoints = []
    high_wind_checkpoints = []
    low_visibility_checkpoints = []
    
    max_rain_probability = 0.0
    
    for pred in predictions:
        weather = pred["weather"]
        is_rainy = pred["prediction"] == 1
        prob = weather["destination_api_rain_probability"]
        
        if prob > max_rain_probability:
            max_rain_probability = prob
            
        if is_rainy:
            rain_checkpoints.append(pred)
            
        # Advisory checks
        if weather.get("destination_wind_speed", 0.0) >= 8.0:
            high_wind_checkpoints.append(pred)
            
        if weather.get("destination_cloud_coverage", 0) >= 85 and is_rainy:
            low_visibility_checkpoints.append(pred)
            
    # Gear Recommendation Logic
    # 1. Continuous moderate/heavy rain -> Raincoat (if >= 2 rainy checkpoints or max prob >= 0.70)
    # 2. Scattered light rain -> Umbrella (if 1 rainy checkpoint or max prob >= 0.35)
    # 3. Clear route -> No protection needed
    
    rainy_count = len(rain_checkpoints)
    
    if rainy_count >= 2 or max_rain_probability >= 0.70:
        recommendation = "RAINCOAT"
        gear_advice = "🚨 High risk of rain along the route. Carry a raincoat (highly recommended for two-wheelers)."
    elif rainy_count == 1 or max_rain_probability >= 0.35:
        recommendation = "UMBRELLA"
        gear_advice = "☔ Scattered light rain expected. Carrying an umbrella should be sufficient for general travel."
    else:
        recommendation = "NONE"
        gear_advice = "☀ Clear route. No rain protection gear needed!"
        
    # Ride Advisory Generation
    advisories = []
    if rain_checkpoints:
        rain_spots = ", ".join([cp["name"] for cp in rain_checkpoints])
        advisories.append(f"⚠️ Wet Roads Alert: Expect slippery riding conditions at {rain_spots}. Reduce your speed.")
        
    if high_wind_checkpoints:
        wind_spots = ", ".join([cp["name"] for cp in high_wind_checkpoints])
        advisories.append(f"💨 High Wind Alert: Strong gusts detected at {wind_spots}. Hold the handlebar firmly.")
        
    if low_visibility_checkpoints:
        vis_spots = ", ".join([cp["name"] for cp in low_visibility_checkpoints])
        advisories.append(f"👁️ Low Visibility Alert: Poor visibility expected at {vis_spots} due to clouds and rain. Turn on headlamps.")
        
    if not advisories:
        advisories.append("✅ Safe travel conditions. Enjoy your ride!")
        
    return {
        "gear_recommendation": recommendation,
        "gear_advice": gear_advice,
        "advisories": advisories,
        "rainy_count": rainy_count,
        "max_rain_probability": max_rain_probability
    }

if __name__ == "__main__":
    print("--- Test Travel Recommendation ---")
    mock_preds = [
        {"name": "Kumbanad", "prediction": 0, "weather": {"destination_api_rain_probability": 0.1, "destination_wind_speed": 3.0, "destination_cloud_coverage": 10}},
        {"name": "Chengannur", "prediction": 1, "weather": {"destination_api_rain_probability": 0.8, "destination_wind_speed": 9.5, "destination_cloud_coverage": 90}},
        {"name": "Mavelikara", "prediction": 1, "weather": {"destination_api_rain_probability": 0.65, "destination_wind_speed": 4.0, "destination_cloud_coverage": 80}}
    ]
    res = generate_route_recommendation(mock_preds)
    print("Gear:", res["gear_recommendation"])
    print("Advice:", res["gear_advice"])
    print("Advisories:")
    for adv in res["advisories"]:
        print("  -", adv)
