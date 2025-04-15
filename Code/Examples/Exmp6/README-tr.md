# Konsol Kiriş Optimizasyonu

Bu projede 5 kısımdan oluşan 3m uzunluğunda bir konsol kirişin, belirli kısıtlamalar altında minimum ağırlık için optimum tasarımı yapılmıştır. Optimizasyon algoritması olarak Simulated Annealing kullanılmıştır.

## Problem Tanımı

- 3m uzunluğunda konsol kiriş, 5 eşit parçaya bölünmüştür
- Her bir kısım içi boş daire profil şeklindedir ve iki tasarım parametresi vardır: 
  - r_i1 (dış yarıçap)
  - r_i2 (iç yarıçap)
- Malzeme olarak S270 çelik kullanılmıştır (E = 210 GPa, σy = 270 MPa)
- Kirişin serbest ucuna dik olarak F = 500 kN'luk bir kuvvet etki etmektedir
- Amaç: Kiriş ağırlığını minimize etmek

### Sınırlayıcılar (Kısıtlamalar)
1. Kirişin uç noktasında maksimum 2 cm yer değiştirme yapmasına izin verilmektedir
2. Her bir kısım için dış yarıçap, iç yarıçaptan büyük olmalıdır
3. Birbiriyle temas eden kısımlardan önce gelenin iç yarıçapı, sonra gelenin dış yarıçapından küçük olmalıdır (kaynak yapılabilirlik şartı)
4. S270 çeliğin akma gerilmesi (270 MPa) aşılmamalıdır

## Yapısal Analiz ve Optimizasyon Yaklaşımı

### Sonlu Elemanlar Analizi
Konsol kirişin yer değiştirme ve gerilme analizi için sonlu elemanlar yöntemi kullanılmıştır:

- Her kiriş parçası Euler-Bernoulli kiriş elemanı olarak modellenmiştir
- Her düğüm noktasında 2 serbestlik derecesi vardır (yer değiştirme ve dönme)
- Global rijitlik matrisi oluşturularak yer değiştirmeler hesaplanmıştır
- Gerilmeler, eğilme momenti ve kesit özellikleri kullanılarak hesaplanmıştır

### Optimizasyon Algoritması
Simulated Annealing algoritması kullanılarak optimum tasarım belirlenmiştir:

- Rastgele arama stratejisi ile lokal optimumlardan kaçınma
- Adaptif adım boyutu kullanarak çözüm uzayının etkili araştırılması
- İşlem sıcaklığının yavaş soğutulması ile daha iyi çözümlerin bulunması
- Sınırlayıcıların etkin kontrolü ile fiziksel olarak uygulanabilir çözümlerin garanti edilmesi

## Optimizasyon Sonuçları

Optimizasyon sonucunda, başlangıç tasarımına kıyasla daha hafif bir kiriş tasarımı elde edilmiştir:

- Başlangıç tasarımı ağırlığı: ~1924 kg
- Optimize edilmiş tasarım ağırlığı: ~939 kg (%51 azalma)

### Optimize Edilmiş Yarıçaplar (cm):

| Segment | Dış Yarıçap (r_o) | İç Yarıçap (r_i) |
|---------|-------------------|------------------|
| 1       | 20.37             | 14.36            |
| 2       | 20.05             | 15.81            |
| 3       | 15.82             | 9.04             |
| 4       | 16.00             | 13.33            |
| 5       | 14.26             | 13.30            |

Optimize edilmiş tasarımda:
- Uç noktadaki yer değiştirme 0.12 cm'dir (izin verilen maksimum değer 2 cm)
- Gerilme sınırlayıcısı aktiftir (0.00 MPa marj)
- Kesit boyutları kaynak edilebilirlik şartını sağlamaktadır

## Görselleştirmeler

### Optimize Edilmiş Kiriş Tasarımı

![Optimize Edilmiş Kiriş](optimized_beam.png)

Bu görselde optimize edilmiş kirişin geometrisi görülmektedir. Mesnet noktasından (sol taraf) uca doğru gidildikçe kesitlerin küçüldüğü gözlenmektedir. Bu durum, eğilme momentinin mesnet noktasında maksimum olup uca doğru azalmasıyla ilişkilidir.

### Deformasyon Şekli

![Deformasyon Şekli](deformed_beam.png)

Bu görsel, konsol kirişin yük altındaki deformasyon şeklini göstermektedir. Kiriş uç noktasında maksimum 2 cm'lik yer değiştirme yapmaktadır. Görselde, her düğüm noktasında oluşan yer değiştirme değerleri de belirtilmiştir.

### Sınırlayıcı Kullanım Oranları

![Sınırlayıcı Kullanım Oranları](constraint_utilization.png)

Bu grafik, optimize edilmiş tasarımda her bir sınırlayıcının ne kadar kullanıldığını göstermektedir:

- **Yer Değiştirme Sınırlayıcısı**: Maksimum yer değiştirme sınırı tamamen kullanılmıştır (%100)
- **Gerilme Sınırlayıcısı**: Her segment için akma gerilmesinin kullanım oranı
- **Yarıçap Oranı**: İç yarıçapın dış yarıçapa oranı
- **Kaynak Edilebilirlik**: Bitişik segmentler arasındaki kaynak şartı kullanım oranı

Grafikten görüldüğü üzere, optimum tasarımda yer değiştirme sınırlayıcısı aktiftir (tamamen kullanılmıştır). Bu durum, optimize edilmiş tasarımın ağırlık minimizasyonu açısından sınıra ulaştığını göstermektedir.

### Optimizasyon Geçmişi

![Optimizasyon Geçmişi](optimization_history.png)

Bu grafik, algoritmanın çalışması süresince en iyi çözümün (minimum ağırlık) nasıl geliştiğini göstermektedir. İlk iterasyonlarda hızlı bir iyileşme olduğu, daha sonra algoritmanın daha küçük iyileştirmeler yaptığı görülmektedir.

## Kullanım

Kodu çalıştırmak için:

```bash
python beam_optimization.py
```

Farklı parametrelerle çalıştırmak için:

```bash
# Farklı kuvvet değeri ile
python beam_optimization.py --force 300000

# Farklı iterasyon sayısı ile
python beam_optimization.py --iterations 10000 --cooling-rate 0.999

# Görselleri oluşturmadan çalıştırmak için
python beam_optimization.py --no-plot
```

## Çıktılar

- `optimization_results.txt`: Optimizasyon sonuçlarını içeren metin dosyası
- `optimization_history.png`: Optimizasyon sürecindeki gelişmeyi gösteren grafik
- `optimized_beam.png`: Optimize edilmiş kirişin geometrisi
- `deformed_beam.png`: Kirişin yük altındaki deformasyon şekli
- `constraint_utilization.png`: Sınırlayıcıların kullanım oranını gösteren grafik

## Sonuç ve Değerlendirme

Bu projede, Simulated Annealing optimizasyon algoritması kullanılarak, kısıtlamalar altında bir konsol kirişin minimum ağırlık için optimum tasarımı gerçekleştirilmiştir. Sonuçlar, başlangıç tasarımına kıyasla %51 daha hafif bir yapı elde edildiğini göstermektedir.

Optimize edilmiş tasarımda, özellikle gerilme sınırlayıcısının tam olarak kullanıldığı (aktif olduğu) görülmektedir. Bu durum, teorik olarak beklenen bir sonuçtur çünkü ağırlık minimizasyonu problemlerinde genellikle en az bir sınırlayıcının aktif olması beklenir.

Kiriş geometrisinin mesnet noktasından uca doğru giderek küçülmesi de, yapısal açıdan beklenen bir sonuçtur. Eğilme momenti mesnet noktasında maksimum olduğundan, bu bölgede daha büyük kesitler oluşmuştur. 