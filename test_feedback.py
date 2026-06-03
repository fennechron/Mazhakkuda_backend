from utils import supabase

# Let's get the latest timestamp that has NULL actual_rain
res = supabase.table('weather_data').select('timestamp, destination, actual_rain').is_('actual_rain', 'null').order('id', desc=True).limit(5).execute()
print("Latest records with NULL actual_rain:")
for r in res.data:
    print(r)
    
if res.data:
    ts = res.data[0]['timestamp']
    print(f"\nQuerying exactly for timestamp '{ts}':")
    res2 = supabase.table('weather_data').select('id, destination, model_prediction').eq('timestamp', ts).execute()
    print("Found rows:", res2.data)
