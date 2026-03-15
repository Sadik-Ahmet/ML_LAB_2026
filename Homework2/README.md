---
# YZM212 Makine Öğrenmesi - 2. Laboratuvar Ödevi

## MLE ile Akıllı Şehir Planlaması (Trafik Yoğunluğu Analizi)

Bu proje, bir ana caddeden geçen araç sayısını **Poisson Dağılımı** kullanarak modellemeyi ve bu modelin parametresini ($\lambda$) **Maksimum Olabilirlik Kestirimi (Maximum Likelihood Estimation - MLE)** yöntemiyle hem analitik hem de sayısal olarak hesaplamayı amaçlamaktadır.

### 📌 Problem Tanımı

Akıllı şehir planlaması kapsamında, bir belediyenin ulaşım departmanı için trafik yoğunluk tahmini yapılması hedeflenmiştir. Dakikada geçen araç sayısının Poisson dağılımına uyduğu varsayılarak, eldeki veriler üzerinden en uygun $\lambda$ (ortalama araç sayısı) parametresi belirlenmiştir.

### 🧪 Yöntem ve Uygulama

#### 1. Teorik Türetme (Analitik Çözüm)

Poisson dağılımı için Olabilirlik (Likelihood) ve Log-Olabilirlik (Log-Likelihood) fonksiyonları türetilmiştir. Fonksiyonun $\lambda$'ya göre türevi alınarak, MLE tahmininin ($\hat{\lambda}_{MLE}$) gözlemlerin **aritmetik ortalamasına** eşit olduğu kanıtlanmıştır.

#### 2. Sayısal (Numerical) MLE

`scipy.optimize.minimize` kütüphanesi kullanılarak Negatif Log-Olabilirlik (NLL) fonksiyonu minimize edilmiştir.

* **Veri Seti:** 14 farklı zaman diliminde kaydedilen trafik verisi.


* **Sonuç:** Analitik ve sayısal yöntemlerin birbirine son derece yakın sonuçlar verdiği (yaklaşık 11.93) doğrulanmıştır.

#### 3. Görselleştirme ve Karşılaştırma

Bulunan $\hat{\lambda}$ değeri ile oluşturulan Poisson Olasılık Kütle Fonksiyonu (PMF), gerçek veri histogramı ile üst üste bindirilerek modelin veriye uyumu analiz edilmiştir.

### ⚠️ Outlier (Aykırı Değer) Analizi

Veri setine hatalı bir gözlem (200 araç) eklenerek MLE'nin bu duruma hassasiyeti test edilmiştir.

* **Bulgu:** Tek bir aykırı değerin $\lambda$ tahminini yaklaşık %106 oranında saptırdığı gözlemlenmiştir.


* **Risk:** Hatalı modelin, belediyeyi gereksiz yol genişletme gibi maliyetli ve yanlış altyapı kararlarına sürükleyebileceği tartışılmıştır.



### 🛠️ Kullanılan Teknolojiler

* **Python 3.x**
* **NumPy:** Veri manipülasyonu.
* **SciPy (optimize & stats):** NLL minimizasyonu ve PMF hesaplamaları.


* **Matplotlib:** Veri görselleştirme.



### 🚀 Nasıl Çalıştırılır?

1. Repoyu klonlayın.
2. Gerekli kütüphaneleri yükleyin: `pip install numpy scipy matplotlib`.
3. `homework2_mle_traffic.ipynb` dosyasını Jupyter Notebook veya VS Code üzerinden çalıştırın.

---
