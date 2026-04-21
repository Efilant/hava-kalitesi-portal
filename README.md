# İstanbul Hava Kalitesi Tahmin Portalı

Merhaba! Bu projede, İstanbul'un farklı bölgelerindeki hava kalitesini (özellikle PM10 seviyelerini) meteorolojik verilerle ilişkilendirerek analiz eden ve gelecek saatler için tahmin yürüten interaktif bir izleme platformu geliştirdim.

## Proje Hakkında
Hava kirliliği, özellikle İstanbul gibi metropollerde yaşam kalitesini doğrudan etkileyen kritik bir konu. Bu çalışmada amacım, sadece mevcut kirlilik verilerini göstermek değil; rüzgar hızı, nem ve sıcaklık gibi meteorolojik faktörlerin kirlilik üzerindeki etkisini modelleyerek kullanıcılara önceden bilgi verebilmekti. 

Portal sayesinde, hava kirliliğinin önümüzdeki saatlerde artıp artmayacağını bilimsel bir yaklaşımla takip edebiliyoruz.

### Neler Sunuyor?
- **Anlık İzleme:** Kadıköy, Beşiktaş, Ümraniye, Sarıyer ve Esenyurt istasyonlarından alınan güncel veriler.
- **Gelecek Tahmini:** Önümüzdeki 3-6 saatlik periyot için hava kalitesi trendleri.
- **Meteorolojik Analiz:** Sıcaklık, nem ve rüzgarın kirlilik üzerindeki etkisini gösteren görsel analizler.
- **Kullanıcı Dostu Tasarım:** Hem teknik verileri merak edenler hem de sadece "hava bugün nasıl?" sorusuna cevap arayanlar için optimize edilmiş arayüz (Koyu/Açık tema desteğiyle).

## Teknik Detaylar
Sistemin arkasında çalışan temel yapı taşları:
- **Makine Öğrenmesi:** Tahmin modeli olarak yüksek doğruluk oranına sahip `Random Forest` algoritmasını kullandım. Model, geçmiş verilerdeki örüntüleri öğrenerek meteorolojik değişimlere göre PM10 seviyesini tahmin eder.
- **Veri Kaynağı:** Gerçek zamanlı meteorolojik veriler için `Open-Meteo API` entegrasyonu sağladım.
- **Arayüz Geliştirme:** Uygulamanın web arayüzünü `Streamlit` kullanarak hazırladım; veri görselleştirmeleri için `Plotly` kütüphanesinden faydalandım.

## Nasıl Çalıştırılır?

Projeyi kendi bilgisayarınızda test etmek isterseniz:

1. **Bağımlılıkları Yükleyin:**
   Terminal üzerinden gerekli Python kütüphanelerini kurun:
   ```bash
   pip install pandas numpy streamlit plotly openmeteo-requests joblib scikit-learn requests-cache retry-requests
   ```

2. **Uygulamayı Başlatın:**
   Proje dizinindeyken portalı şu komutla çalıştırabilirsiniz:
   ```bash
   streamlit run app.py
   ```

## Dosya Yapısı
- `app.py`: Portalın arayüzünü ve tahmin mantığını yöneten ana dosya.
- `train_model.py`: Veri işleme ve makine öğrenmesi modelinin eğitildiği bölüm.
- `neutron_model.joblib`: Sistemin kalbi olan, önceden eğitilmiş tahmin modeli.
- `fetch_meteorology.py`: Hava durumu verilerini API üzerinden çeken modül.

---
**Geliştirici:** Elif Altun  
*Bu proje, İstanbul'un hava kalitesini daha görünür kılmak ve teknolojiyle çevre farkındalığını birleştirmek amacıyla hazırlanmıştır.*
