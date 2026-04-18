# train_model.py - İleri Seviye Hava Kalitesi Tahmin Modeli
# Fizik Tabanlı Özellik Mühendisliği ve Sinyal İşleme Entegrasyonu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
import joblib
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="darkgrid")

print("🚀 Eğitim ve Analiz Süreci Başlatılıyor...")

# 1. VERİ YÜKLEME VE BİRLEŞTİRME
try:
    df_hava = pd.read_csv('hava_kalitesi_veriseti_20260418.csv')
    df_meteo = pd.read_csv('meteoroloji_veriseti.csv')
    
    df_hava['ReadTime'] = pd.to_datetime(df_hava['ReadTime'])
    df_meteo['Zaman'] = pd.to_datetime(df_meteo['Zaman'])
    
    df_hava = df_hava.sort_values('ReadTime')
    df_meteo = df_meteo.sort_values('Zaman')
    
    # asof merge: Zaman serilerini en yakın saat bazında birleştirir
    df = pd.merge_asof(df_hava, df_meteo, left_on='ReadTime', right_on='Zaman', direction='nearest')
    print(f"✅ Veri birleştirme başarılı. Satır sayısı: {len(df)}")
except Exception as e:
    print(f"❌ Veri Yükleme Hatası: {e}"); exit()

# 2. SİNYAL İŞLEME VE GÜRÜLTÜ FİLTRELEME
for col in ['PM10', 'NO2', 'SO2', 'O3']:
    if col in df.columns:
        df[f'{col}_Clean'] = savgol_filter(df[col].ffill().bfill(), window_length=7, polyorder=2)

# 3. ÖZELLİK MÜHENDİSLİĞİ (Fizik Temelli Parametreler)
df['Hour'] = df['ReadTime'].dt.hour
df['DayOfWeek'] = df['ReadTime'].dt.dayofweek

df['Hygroscopic_Index'] = df['Nem'] * df['PM10_Clean'].shift(1)
df['Dispersion_Index'] = df['PM10_Clean'].shift(1) / (df['RuzgarHizi'] + 0.5)
df['Thermal_Stability'] = df['Sicaklik'].diff() * df['Nem']
df['Ventilation_Score'] = df['RuzgarHizi'] * df['Sicaklik'].rolling(window=3).std()

df['PM10_Velocity'] = df['PM10_Clean'].diff()
df['PM10_Acceleration'] = df['PM10_Velocity'].diff()
df['Wind_Trend'] = df['RuzgarHizi'].diff()

for lag in [1, 3, 12, 24]:
    df[f'PM10_Lag{lag}'] = df['PM10_Clean'].shift(lag)

features = [
    'SO2_Clean', 'NO2_Clean', 'O3_Clean', 'Hour', 'DayOfWeek', 
    'Hygroscopic_Index', 'Dispersion_Index', 'Thermal_Stability', 'Ventilation_Score',
    'PM10_Velocity', 'PM10_Acceleration', 'Wind_Trend',
    'PM10_Lag1', 'PM10_Lag3', 'PM10_Lag24', 
    'Sicaklik', 'Nem', 'RuzgarHizi'
]

# 4. HATA ANALİZİ VE ÖZEL ÖRNEKLEM AĞIRLIKLANDIRMA
df['Target'] = np.log1p(df['PM10'].ffill().bfill())
median_pm = df['PM10'].median()
max_pm = df['PM10'].max()
df['Sample_Weight'] = 1.0 + 2.0 * (df['PM10'] - median_pm).abs() / max_pm

df = df.dropna(subset=features + ['Target', 'Sample_Weight']).reset_index(drop=True)
print(f"✅ Eğitim seti hazırlandı. Kullanılabilir satır: {len(df)}")

X = df[features]
y = df['Target']
weights = df['Sample_Weight']

split = int(len(df) * 0.85)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
w_train = weights[:split]
y_test_real = np.expm1(y_test)

# 5. MODEL EĞİTİMİ (Random Forest Regressor)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000, max_depth=30, min_samples_split=4, min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)

print("🤖 Model eğitiliyor, lütfen bekleyin...")
model.fit(X_train, y_train, sample_weight=w_train)

# 6. PERFORMANS METRİKLERİ
y_pred_log = model.predict(X_test)
y_pred_real = np.expm1(y_pred_log)
r2 = r2_score(y_test_real, y_pred_real)
mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

print(f"\n" + "="*40)
print(f"📊 MODEL PERFORMANS RAPORU")
print(f"="*40)
print(f"R2 Skor: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"="*40)

# 7. GÖRSEL ANALİZ VE SONUÇLAR
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(df['ReadTime'].iloc[split:].values, y_test_real.values, label='Gerçek Ölçüm (İBB)', color='#ecf0f1', alpha=0.6, linewidth=1.5)
ax1.plot(df['ReadTime'].iloc[split:].values, y_pred_real, label='Tahmin Değeri', color='#00d2ff', linewidth=2.5)
ax1.set_title(f"Hava Kalitesi Zaman Serisi Tahmin Analizi (R2: {r2:.3f})", fontsize=16, pad=20, color='white')
ax1.fill_between(df['ReadTime'].iloc[split:].values, y_test_real, y_pred_real, color='#00d2ff', alpha=0.1)
ax1.legend(loc='upper right', frameon=True, facecolor='#2c3e50')
ax1.set_ylabel("PM10 Konsantrasyonu (µg/m³)")

importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
sns.barplot(data=importance.head(10), x='importance', y='feature', ax=ax2, palette='Blues_r')
ax2.set_title("Tahminde En Etkili Meteorolojik ve Fiziksel Parametreler", fontsize=14, color='white')

plt.tight_layout()
plt.savefig('tahmin_analiz_sonuclari.png')
print("📊 Analiz grafiği 'tahmin_analiz_sonuclari.png' olarak kaydedildi.")

# 8. MODEL KAYDETME (Application Entegrasyonu İçin)
print("💾 Model ve özellik seti dışa aktarılıyor...")
model_data = {
    'model': model,
    'features': features,
    'r2_score': r2,
    'last_trained_r2': r2,
    'timestamp': pd.Timestamp.now().isoformat()
}
joblib.dump(model_data, 'neutron_model.joblib')
print("✅ 'neutron_model.joblib' başarıyla kaydedildi.")

# plt.show() # Web ortamında engel olmaması için yorum satırına alındı
