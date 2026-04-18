# İstanbul Hava Kalitesi İzleme Portalı

Bu proje, İstanbul genelindeki hava kalitesi verilerini (PM10, NO2, SO2, O3) anlık meteorolojik faktörlerle (Sıcaklık, Nem, Rüzgar Hızı) analiz ederek gelecek tahminleri sunan bir izleme platformudur.

## 🏙️ İnteraktif Kullanıcı Paneli
Portal, kullanıcıların İstanbul'daki çeşitli istasyonları seçerek anlık kirlilik seviyelerini ve önümüzdeki 6 saatlik trendleri takip etmesine olanak tanır.

### Temel Özellikler:
*   **Otomatik Tema Desteği:** Sistem ayarlarınıza göre Açık (Light) veya Koyu (Dark) mod desteği.
*   **Anlık Tahminler:** Meteorolojik değişimlerin hava kalitesine etkisini saniyeler içinde hesaplar.
*   **Bölgesel İzleme:** Kadıköy, Beşiktaş gibi farklı istasyonlar arasında geçiş yapabilme.
*   **Anlaşılır Metrikler:** Teknik terimlerden arındırılmış, herkesin anlayabileceği "Durum" bilgisi.

## Teknik Altyapı
*   **Veri Kaynağı:** Open-Meteo API (Anlık hava durumu).
*   **Tahmin Modeli:** Random Forest tabanlı yüksek hassasiyetli regresyon modeli.
*   **Yazılım:** Streamlit framework üzerine kurulu Python uygulaması.

## Kurulum ve Çalıştırma

### 1. Kütüphaneleri Yükleyin
```bash
pip install pandas numpy streamlit plotly openmeteo-requests joblib scikit-learn scipy
```

### 2. Portal'ı Başlatın
```bash
streamlit run app.py
```

## Dosya Yapısı
*   `app.py`: Ana portal uygulaması.
*   `train_model.py`: Arka planda çalışan analiz ve model eğitim scripti.
*   `neutron_model.joblib`: Sistemin temelini oluşturan eğitimli model dosyası.
*   `README.md`: Kullanım ve teknik detaylar dökümanı.
