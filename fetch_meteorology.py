# fetch_meteorology.py
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Ümraniye Koordinatları ve 2026 Arşivi
params = {
	"latitude": 41.0343,
	"longitude": 29.1144,
	"start_date": "2026-01-01",
	"end_date": "2026-04-18",
	"hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m"],
	"timezone": "Europe/Istanbul"
}

try:
    print("Open-Meteo üzerinden meteoroloji verileri çekiliyor...")
    responses = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
    response = responses[0]
    hourly = response.Hourly()
    
    hourly_data = {"Zaman": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["Sicaklik"] = hourly.Variables(0).ValuesAsNumpy()
    hourly_data["Nem"] = hourly.Variables(1).ValuesAsNumpy()
    hourly_data["RuzgarHizi"] = hourly.Variables(2).ValuesAsNumpy()
    hourly_data["RuzgarYonu"] = hourly.Variables(3).ValuesAsNumpy()

    df_meteo = pd.DataFrame(data = hourly_data)
    df_meteo['Zaman'] = df_meteo['Zaman'].dt.tz_convert('Europe/Istanbul').dt.tz_localize(None)
    
    df_meteo.to_csv('meteoroloji_veriseti.csv', index=False)
    print("✅ Başarılı! 'meteoroloji_veriseti.csv' oluşturuldu.")
except Exception as e:
    print(f"❌ Hata: {e}")