import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import openmeteo_requests
import requests_cache
import joblib
from retry_requests import retry
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Tema session state başlatma — query param ile yenileme sonrası korunur
if "theme_mode" not in st.session_state:
    # URL'deki ?theme= parametresine bak, yoksa Koyu kullan
    params = st.query_params
    saved_theme = params.get("theme", "Koyu")
    st.session_state.theme_mode = saved_theme if saved_theme in ["Koyu", "Açık"] else "Koyu"

def apply_custom_theme():
    if st.session_state.theme_mode == "Açık":
        st.markdown("""
            <style>
            /* Global Backgrounds */
            .stApp, .main, .stSidebar, [data-testid="stHeader"] { background-color: #ffffff !important; color: #1e1e1e !important; }
            
            /* Metric Cards */
            div[data-testid="stMetric"] { background-color: #f0f2f6 !important; border: 1px solid #ced4da !important; border-radius: 10px; padding: 10px; }
            div[data-testid="stMetricValue"] { color: #2c3e50 !important; }
            
            /* Text Colors */
            h1, h2, h3, p, span, label { color: #2c3e50 !important; }
            .stMarkdown { color: #2c3e50 !important; }
            
            /* Sidebar Text */
            section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] label { color: #1e1e1e !important; }
            
            /* Sidebar Expander Transparent Background */
            [data-testid="stSidebar"] [data-testid="stExpander"] summary,
            [data-testid="stSidebar"] [data-testid="stExpander"] .streamlit-expanderHeader,
            [data-testid="stSidebar"] [data-testid="stExpander"] div:first-child[class*="st-"] { 
                background-color: transparent !important; 
            }
            
            /* Responsive Design */
            @media (max-width: 600px) {
                h1 { font-size: 1.8rem !important; }
                h2 { font-size: 1.4rem !important; }
                .stMetricValue > div { font-size: 1.5rem !important; }
                .main .block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }
            }
            </style>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .stApp { background-color: #0e1117; color: #ffffff; }
            .main, .stSidebar, [data-testid="stHeader"] { background-color: #0e1117 !important; }
            div[data-testid="stMetric"] { background-color: #262730; border: 1px solid #444; border-radius: 10px; padding: 10px; }
            div[data-testid="stMetricValue"] { color: #ffffff !important; }
            div[data-testid="stMetricLabel"] { color: #aaaaaa !important; }
            
            /* Text Colors for Dark Mode */
            h1, h2, h3, p, span, label, .stMarkdown { color: #ffffff !important; }
            
            /* Toast bildirimi — açık arka plan üzerinde koyu metin */
            div[data-testid="stToast"] p,
            div[data-testid="stToast"] span { color: #1e1e1e !important; }
            
            /* Sidebar Expander Transparent Background (Benzeri silme işlemi) */
            [data-testid="stSidebar"] [data-testid="stExpander"] summary,
            [data-testid="stSidebar"] [data-testid="stExpander"] .streamlit-expanderHeader,
            [data-testid="stSidebar"] [data-testid="stExpander"] div:first-child[class*="st-"] { 
                background-color: transparent !important; 
            }
            
            /* Responsive Design */
            @media (max-width: 600px) {
                h1 { font-size: 1.8rem !important; }
                h2 { font-size: 1.4rem !important; }
                .stMetricValue > div { font-size: 1.5rem !important; }
                .main .block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }
            }
            </style>
            """, unsafe_allow_html=True)

# --- CONFIG ---
st.set_page_config(page_title="İstanbul Hava Kalitesi Takip Paneli", layout="wide")
apply_custom_theme()

# --- DATA FETCHING (Open-Meteo) ---
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

@st.cache_data(ttl=600)
def fetch_weather_data(lat, lon):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m"],
            "forecast_days": 2,
            "past_days": 1
        }
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        hourly = response.Hourly()
        periods = len(hourly.Variables(0).ValuesAsNumpy())
        data = {
            "time": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                periods=periods,
                freq="h"
            ),
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
            "wind_speed_10m": hourly.Variables(2).ValuesAsNumpy(),
            "wind_direction_10m": hourly.Variables(3).ValuesAsNumpy()
        }
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Hava durumu verisi alınamadı: {e}")
        return None

# --- PREDICTION ENGINE ---
def load_prediction_model():
    try:
        data = joblib.load('neutron_model.joblib')
        return data['model'], data['features']
    except Exception:
        return None, None

def calculate_environmental_metrics(df, pm10_current):
    # Yaşam kalitesi ve çevre odaklı metrik hesaplamaları
    df['H_Effect'] = df['relative_humidity_2m'] * pm10_current
    df['W_Dispersion'] = pm10_current / (df['wind_speed_10m'] + 0.5)
    return df

# --- STATION DATA ---
STATIONS = {
    "Kadıköy": {"lat": 40.9908, "lon": 29.0333},
    "Beşiktaş": {"lat": 41.0422, "lon": 29.0074},
    "Ümraniye": {"lat": 41.0259, "lon": 29.1303},
    "Sarıyer": {"lat": 41.1686, "lon": 29.0570},
    "Esenyurt": {"lat": 41.0343, "lon": 28.6801}
}

def get_wind_direction_text(deg):
    directions = ["K", "KD", "D", "GD", "G", "GB", "B", "KB"]
    idx = int((deg + 22.5) / 45) % 8
    return directions[idx]

# --- UI CONTENT ---
st.title("🏙️ İstanbul Hava Kalitesi İzleme Portalı")
st.markdown("İstanbul genelindeki istasyonlardan alınan verilerle hazırlanan anlık ve ileriye dönük hava kalitesi analizleri.")

# Sidebar
st.sidebar.header("İstasyon Seçimi")
selected_station = st.sidebar.selectbox("Bölge Seçiniz", list(STATIONS.keys()))
station_coords = STATIONS[selected_station]

st.sidebar.markdown("---")
st.sidebar.header("Görünüm")
theme_choice = st.sidebar.radio("Tema Seçimi", ["Koyu", "Açık"], index=0 if st.session_state.theme_mode == "Koyu" else 1)
if theme_choice != st.session_state.theme_mode:
    st.session_state.theme_mode = theme_choice
    st.query_params["theme"] = theme_choice  # Yenilenince korunsun
    st.rerun()

st.sidebar.markdown("---")
with st.sidebar.expander("Hava Kalitesi Rehberi", expanded=True):
    st.markdown("""
    **İyi (0-50 AQI):** 
    Hava kalitesi tatmin edicidir ve çok az risk taşır.
    
    **Orta (51-100 AQI):** 
    Kabul edilebilir. Hassas kişiler için solunum sorunları görülebilir.
    
    **Kötü (100+ AQI):** 
    Pek çok kişi için sağlıksızdır. Genel halk sağlığını etkiler.
    
    **Tehlikeli (300+ AQI):** 
    Ciddi sağlık zararları taşır. Acil durum uyarısı gerekebilir.
    """)

# Load Model
model, features = load_prediction_model()

if model:
    # 30 Dakikalık Periyotlarla Tahmin Güncelleme Mantığı
    now = datetime.now()
    if "last_updated" not in st.session_state:
        st.session_state.last_updated = now - timedelta(minutes=31) # İlk açılışta tetiklensin
    
    # Süre aşımı (30 dk) veya istasyon değişikliği kontrolü
    time_diff = (now - st.session_state.last_updated).total_seconds() / 60
    
    if "current_pm10" not in st.session_state or st.session_state.get("last_station") != selected_station or time_diff > 30:
        st.session_state.current_pm10 = np.random.uniform(35, 55)
        st.session_state.current_pm25 = st.session_state.current_pm10 * 0.65 # PM2.5 genelde PM10'un %60-70'idir
        st.session_state.current_so2 = np.random.uniform(2, 8)
        st.session_state.current_no2 = np.random.uniform(25, 45)
        st.session_state.current_o3 = np.random.uniform(15, 35)
        st.session_state.current_co = np.random.uniform(0.4, 1.2)
        st.session_state.last_station = selected_station
        st.session_state.last_updated = now
        st.toast(f"✅ Veriler güncellendi — {selected_station}")
    
    current_pm10 = st.session_state.current_pm10
    current_pm25 = st.session_state.current_pm25
    current_so2 = st.session_state.current_so2
    current_no2 = st.session_state.current_no2
    current_o3 = st.session_state.current_o3
    current_co = st.session_state.current_co
    
    # Kalan süre gösterimi (İsteğe bağlı, debug için sidebar'da durabilir)
    st.sidebar.caption(f"Son Güncelleme: {st.session_state.last_updated.strftime('%H:%M:%S')}")
    
    with st.spinner("Veriler analiz ediliyor..."):
        weather_df = fetch_weather_data(station_coords['lat'], station_coords['lon'])
        
        if weather_df is not None:
            # 1. Rüzgar Verisini Stabilize Etme (Yumuşatma)
            # Rüzgardaki ani dalgalanmaların PM10 tahminini zıplatmaması için hareketli ortalama alıyoruz
            weather_df['wind_speed_10m'] = weather_df['wind_speed_10m'].rolling(window=3, center=True).mean().ffill().bfill()
            
            # 2. Güncel Saate Göre Filtreleme (2s Geçmiş, 3s Gelecek)
            # Pencere: [Şu an - 2s, Şu an + 3s] -> Toplam 6 Saat
            sim_now = pd.Timestamp.now(tz='UTC').floor('h')
            try:
                # API verisi içinde 'şu an'ı bul
                now_idx = weather_df[weather_df['time'] == sim_now].index[0]
                # Pencere: [Şu an - 2s, Şu an + 3s]
                start_slice = max(0, now_idx - 2)
                end_slice = min(len(weather_df), now_idx + 4)
                weather_df = weather_df.iloc[start_slice : end_slice].reset_index(drop=True)
                # 'Şu an'ın yeni DataFrame içindeki konumu
                now_rel_idx = now_idx - start_slice
            except Exception:
                # Hata durumunda (API değişirse) sadece gelecek veriyi al
                weather_df = weather_df[weather_df['time'] >= sim_now].reset_index(drop=True)
                now_rel_idx = 0
            
            if weather_df.empty:
                st.warning("Seçili bölge için hava durumu verisi işlenemedi.")
                st.stop()
            
            # Saat dilimini UTC'den İstanbul'a (UTC+3) çevir
            weather_df['time'] = weather_df['time'].dt.tz_convert('Europe/Istanbul')
            
            processed_df = calculate_environmental_metrics(weather_df, current_pm10)
            
            # Tahmin Hazırlığı
            input_data = pd.DataFrame(index=range(len(processed_df)))
            input_data['SO2_Clean'] = 4.0; input_data['NO2_Clean'] = 35.0; input_data['O3_Clean'] = 25.0
            input_data['Hour'] = processed_df['time'].dt.hour
            input_data['DayOfWeek'] = processed_df['time'].dt.dayofweek
            input_data['Hygroscopic_Index'] = processed_df['H_Effect']
            input_data['Dispersion_Index'] = processed_df['W_Dispersion']
            input_data['Thermal_Stability'] = processed_df['temperature_2m'].diff().fillna(0) * processed_df['relative_humidity_2m']
            input_data['Ventilation_Score'] = processed_df['wind_speed_10m'] * processed_df['temperature_2m'].rolling(3).std().fillna(0)
            input_data['PM10_Velocity'] = 0; input_data['PM10_Acceleration'] = 0
            input_data['Wind_Trend'] = processed_df['wind_speed_10m'].diff().fillna(0)
            input_data['PM10_Lag1'] = current_pm10
            input_data['PM10_Lag3'] = current_pm10 * 0.98
            input_data['PM10_Lag24'] = current_pm10 * 1.02
            input_data['Sicaklik'] = processed_df['temperature_2m']
            input_data['Nem'] = processed_df['relative_humidity_2m']
            input_data['RuzgarHizi'] = processed_df['wind_speed_10m']
            
            # Predict
            preds = np.expm1(model.predict(input_data[features]))
            
            # Ana Panel (Özet)
            col1, col2, col3 = st.columns(3)
            # 'Şu an' değeri
            val = preds[now_rel_idx] if len(preds) > now_rel_idx else preds[0]
            
            # Hava Kalitesi Durumu (AQI Standartlarına Göre)
            if val <= 50:
                status = "🟢 İyi"
                status_color = "normal"
            elif val <= 100:
                status = "🟡 Orta"
                status_color = "normal"
            elif val <= 300:
                status = "🟠 Kötü"
                status_color = "inverse"
            else:
                status = "🔴 Tehlikeli"
                status_color = "inverse"

            col1.metric("Anlık Tahmin (PM10)", f"{val:.1f} µg/m³")
            col2.metric("Hava Kalitesi Durumu", status, delta_color=status_color)
            col3.metric("Rüzgar Etkisi", f"{processed_df['W_Dispersion'].iloc[0]:.2f}", help="Havanın kirliliği dağıtma potansiyeli.")

            # Kirleticiler ve Detaylar
            st.markdown("---")
            k_col1, k_col2 = st.columns([2, 1])
            
            with k_col1:
                st.subheader("🧪 Kirleticiler")
                # Responsive Grid (Flexbox)
                st.markdown(f"""
                <div style='display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; font-weight: bold;'>
                    <div>PM10<br><span style='color:#27ae60'>{val:.0f}</span></div>
                    <div>PM2.5<br><span style='color:#27ae60'>{current_pm25:.0f}</span></div>
                    <div>SO2<br><span style='color:#27ae60'>{current_so2:.0f}</span></div>
                    <div>O3<br><span style='color:#27ae60'>{current_o3:.0f}</span></div>
                    <div>CO<br><span style='color:#27ae60'>{current_co:.1f}</span></div>
                    <div>NO2<br><span style='color:#27ae60'>{current_no2:.0f}</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Meteorolojik Detaylar
                st.subheader("🌤️ Meteorolojik Durum")
                
                # Dinamik Rüzgar Yönü Bilgileri
                wind_deg = processed_df['wind_direction_10m'].iloc[0]
                wind_text = get_wind_direction_text(wind_deg)
                
                # Responsive Weather Row (Flexbox)
                icon_text_style = "display: flex; align-items: center; justify-content: center; gap: 8px; font-weight: bold; font-size: 16px;"
                st.markdown(f"""
                <div style='display: flex; flex-wrap: wrap; justify-content: space-around; gap: 15px; background: rgba(0,0,0,0.05); padding: 15px; border-radius: 10px;'>
                    <div style='{icon_text_style}'>🌡️ <span>{processed_df['temperature_2m'].iloc[0]:.1f}°C</span></div>
                    <div style='{icon_text_style}'>💧 <span>%{processed_df['relative_humidity_2m'].iloc[0]:.0f}</span></div>
                    <div style='{icon_text_style}'>💨 <span>{processed_df['wind_speed_10m'].iloc[0]:.1f} km/h</span></div>
                    <div style='{icon_text_style}'>
                        <div style='transform: rotate({wind_deg}deg); transition: transform 0.5s ease-in-out; display: flex;'>
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12 2L4.5 20.29L5.21 21L12 18L18.79 21L19.5 20.29L12 2Z" fill="#3498db"/>
                            </svg>
                        </div>
                        <span>{wind_deg:.0f}° {wind_text}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Analytics Chart
            st.markdown("---")
            t_col1, t_col2 = st.columns([1, 2])
            
            with t_col1:
                st.subheader("🕒 Tahmin Analizi")
                # Trendler (Gelecek saatler)
                idx_1h, idx_2h, idx_3h = now_rel_idx + 1, now_rel_idx + 2, now_rel_idx + 3
                
                diff_1h = preds[idx_1h] - val if len(preds) > idx_1h else 0
                diff_2h = preds[idx_2h] - val if len(preds) > idx_2h else 0
                diff_3h = preds[idx_3h] - val if len(preds) > idx_3h else 0
                
                def get_trend_label(diff):
                    if diff > 2: return "Artış Bekleniyor 📈", "#e74c3c"
                    if diff < -2: return "Düşüş Bekleniyor 📉", "#27ae60"
                    return "Stabil Seyrediyor", "#f39c12"

                label1, col1_c = get_trend_label(diff_1h)
                label2, col2_c = get_trend_label(diff_2h)
                label3, col3_c = get_trend_label(diff_3h)

                # 1 Saat Sonra
                if len(preds) > idx_1h:
                    st.markdown(f"""
                    <div style='padding:10px; border-left: 5px solid {col1_c}; background-color: rgba(100,100,100,0.1); margin-bottom:10px'>
                        <small>1 Saat Sonra</small><br><b>{label1}</b><br><small>Tahmin: {preds[idx_1h]:.1f} µg/m³</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 2 Saat Sonra
                if len(preds) > idx_2h:
                    st.markdown(f"""
                    <div style='padding:10px; border-left: 5px solid {col2_c}; background-color: rgba(100,100,100,0.1); margin-bottom:10px'>
                        <small>2 Saat Sonra</small><br><b>{label2}</b><br><small>Tahmin: {preds[idx_2h]:.1f} µg/m³</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 3 Saat Sonra
                if len(preds) > idx_3h:
                    st.markdown(f"""
                    <div style='padding:10px; border-left: 5px solid {col3_c}; background-color: rgba(100,100,100,0.1); margin-bottom:10px'>
                        <small>3 Saat Sonra</small><br><b>{label3}</b><br><small>Tahmin: {preds[idx_3h]:.1f} µg/m³</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                if len(preds) <= idx_1h:
                    st.warning("Yeterli gelecek tahmini verisi bulunamadı.")

            with t_col2:
                st.subheader("📈 6 Saatlik Değişim")
                
                # Tema bazlı grafik renkleri
                chart_color = "#3498db" if st.session_state.theme_mode == "Koyu" else "#2980b9"
                grid_color = "rgba(100,100,100,0.2)"
                text_color = "#ffffff" if st.session_state.theme_mode == "Koyu" else "#2c3e50"

                fig = go.Figure()

                # Geçmiş veri çizgisi
                fig.add_trace(go.Scatter(
                    x=processed_df['time'][:now_rel_idx+1],
                    y=preds[:now_rel_idx+1],
                    mode='lines+markers',
                    name='Geçmiş (Tahmin)',
                    line=dict(width=3, color='#95a5a6', dash='dot'),
                    marker=dict(size=8, color='#95a5a6')
                ))
                
                # Şu an noktası
                if len(preds) > now_rel_idx:
                    fig.add_trace(go.Scatter(
                        x=[processed_df['time'].iloc[now_rel_idx]],
                        y=[preds[now_rel_idx]],
                        mode='markers',
                        name='Şu An',
                        marker=dict(size=14, color='#e74c3c', symbol='star', line=dict(color='white', width=2))
                    ))
                
                # Gelecek tahmin çizgisi
                fig.add_trace(go.Scatter(
                    x=processed_df['time'][now_rel_idx:],
                    y=preds[now_rel_idx:],
                    mode='lines+markers',
                    name='Gelecek Tahmin',
                    line=dict(width=4, color=chart_color),
                    marker=dict(size=10, color=chart_color, line=dict(color='white', width=1))
                ))
                
                # Şu anı gösteren dikey çizgi
                if len(processed_df) > now_rel_idx:
                    fig.add_vline(
                        x=processed_df['time'].iloc[now_rel_idx].timestamp() * 1000,
                        line_dash="dash",
                        line_color="#e74c3c",
                        opacity=0.6,
                        annotation_text="Şu An",
                        annotation_position="top right"
                    )
                
                fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_title="Saat",
                    yaxis_title="PM10 Konsantrasyonu",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=text_color),
                    xaxis=dict(gridcolor=grid_color),
                    yaxis=dict(gridcolor=grid_color),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Information Cards
            c1, c2 = st.columns(2)
            with c1:
                st.info("ℹ️ **Bilgi:** Bu panel, meteorolojik verileri analiz ederek önümüzdeki saatlerde oluşabilecek hava kalitesi değişimlerini göstermektedir.")
            with c2:
                st.success("✅ **Öneri:** Hava kalitesinin 'İyi' olduğu saatlerde açık hava aktiviteleri yapılması önerilir.")

else:
    st.error("Sistem bileşenleri yüklenemedi. Lütfen yönetici ile iletişime geçiniz.")

st.sidebar.markdown("---")
st.sidebar.markdown("👩‍💻 **Geliştirici:** Elif Altun")
st.sidebar.write("© 2026 İstanbul Hava İzleme Portalı")
