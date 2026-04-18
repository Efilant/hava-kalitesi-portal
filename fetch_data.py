import requests
import pandas as pd

def fetch_ibb_meteorology():
    # AWOS İstasyonları Veri Seti ID'si
    resource_id = "a8917593-cc65-4348-ad05-b92e6848ecdf"
    url = f"https://data.ibb.gov.tr/api/3/action/datastore_search?resource_id={resource_id}&limit=10000"

    print("İBB Açık Veri Portalı üzerinden meteoroloji verileri çekiliyor...")
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # CKAN API yapısında veriler 'result' -> 'records' içindedir
            records = data['result']['records']
            
            if not records:
                print("Veri bulunamadı.")
                return None
                
            df = pd.DataFrame(records)
            
            # CSV olarak kaydet
            df.to_csv('meteoroloji_veriseti.csv', index=False)
            print(f"✅ Başarılı! {len(df)} adet meteorolojik kayıt çekildi.")
            print("\nSütun İsimleri:", df.columns.tolist())
            print(df.head())
            return df
        else:
            print(f"Hata: {response.status_code}")
            return None
    except Exception as e:
        print(f"Bağlantı hatası oluştu: {e}")
        return None

if __name__ == "__main__":
    fetch_ibb_meteorology()