# Yapısal Optimizasyon Örnekleri

Bu dizin, çeşitli yapısal optimizasyon tekniklerini gösteren pratik örnekleri içermektedir. Her örnek, optimizasyon problemlerinin farklı yönlerine ve çözümlerine odaklanmaktadır.

## Örnek Yapısı

Örnekler numaralandırılmış klasörlerde (Exmp1'den Exmp7'ye) düzenlenmiştir ve her biri belirli bir optimizasyon problemi ve çözümünü içermektedir. Her örnek şunları içerir:
- Uygulama kodu
- Dokümantasyon (README.md)
- Görselleştirme dosyaları
- Sonuçlar ve analiz

## Örnek Açıklamaları

### Exmp1: Ackley Fonksiyonu Optimizasyonu
- **Amaç**: Farklı algoritmalar kullanarak Ackley fonksiyonunun optimizasyonunu gösterir
- **Dosyalar**:
  - `ackley_optimization.py`: Ana uygulama
  - `pseudo.txt`: Algoritma sözde kodu
  - Optimizasyon sürecinin görselleştirmeleri
- **Önemli Özellikler**: Bir kıyaslama fonksiyonu için çeşitli optimizasyon algoritmalarının uygulanması

### Exmp2: Optimizasyon Yöntemlerinin Karşılaştırılması
- **Amaç**: Farklı klasik optimizasyon yöntemlerini karşılaştırır
- **Dosyalar**:
  - `optimization_methods.py`: Çeşitli yöntemlerin uygulanması
  - Yakınsama gösteren çoklu görselleştirme dosyaları
- **Önemli Özellikler**: Gradyan inişi, Newton yöntemi ve diğer klasik yaklaşımların karşılaştırılması

### Exmp3: Tekil ve Popülasyon Temelli Optimizasyon
- **Amaç**: Tekil çözüm ve popülasyon temelli optimizasyon yaklaşımlarını karşılaştırır
- **Dosyalar**:
  - `s_vs_p.py`: Ana uygulama
  - Algoritma performansının görselleştirmeleri
- **Önemli Özellikler**: TLBO ve Tabu Arama algoritmalarının karşılaştırılması

### Exmp4: Gezgin Satıcı Problemi
- **Amaç**: Farklı algoritmalar kullanarak Gezgin Satıcı Problemini çözer
- **Dosyalar**:
  - `tsp.py`: Çeşitli TSP algoritmalarının uygulanması
  - Rota görselleştirmeleri
- **Önemli Özellikler**: En yakın komşu, 2-opt ve tavlama benzetimi uygulaması

### Exmp5: Çok Amaçlı Optimizasyon Çerçevesi
- **Amaç**: Çok amaçlı optimizasyon problemleri için bir çerçeve sunar
- **Dosyalar**:
  - `main.py`: Temel uygulama
  - `algorithms/`: Çeşitli optimizasyon algoritmaları
  - `utils/`: Yardımcı fonksiyonlar
- **Önemli Özellikler**: Çok amaçlı optimizasyon için modüler çerçeve

### Exmp6: Kiriş Optimizasyonu
- **Amaç**: Bir konsol kiriş yapısını optimize eder
- **Dosyalar**:
  - `beam_optimization.py`: Ana uygulama
  - Çeşitli görselleştirme dosyaları
- **Önemli Özellikler**: Kısıtlamalarla yapısal optimizasyon

### Exmp7: Çelik Çerçeve Optimizasyonu
- **Amaç**: SAP2000 OAPI kullanarak bir çelik çerçeve yapısını optimize eder
- **Dosyalar**:
  - `sim_an_sap2000.py`: SAP2000 kullanılarak uygulama
  - Model dosyaları ve optimizasyon sonuçları
- **Önemli Özellikler**: Ticari yapısal analiz yazılımı ile entegrasyon

## Kullanım

Her örnek bağımsız olarak çalıştırılabilir. Lütfen her örnek dizinindeki bireysel README dosyalarına bakarak:
- Gerekli bağımlılıklar
- Kodu çalıştırma yöntemi
- Beklenen çıktılar
- Görselleştirme seçenekleri
hakkında özel talimatlar için başvurun.

## Bağımlılıklar

Örnekler arasında ortak bağımlılıklar şunlardır:
- Python 3.x
- NumPy
- SciPy
- Matplotlib
- Pandas

Belirli örnekler ek bağımlılıklar gerektirebilir. Lütfen detaylı gereksinimler için bireysel örneğin README dosyasını kontrol edin.