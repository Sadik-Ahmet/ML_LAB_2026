# YZM212 Makine Ogrenmesi Dersi 4. Laboratuvar Odevi
## Uzak Bir Galaksinin Parlaklik Analizi

**Ogrenci:** [Adinizi / Iletisim Bilgilerinizi Buraya Giriniz]

### Problem Tanimi
Astronomide laboratuvar ortami olmadigi icin gozlem verilerinden model parametrelerini (ornegin bir yildizin kutlesi veya yaydigi foton miktari) tahmin etmek zordur. Bu projede, gurultulu uzak galaksi gozlem verilerini kullanarak gercek parlaklik ($\mu$) ve yapi/olcum belirsizligi ($\sigma$) parametrelerini Bayesyen istatistik yontemleri (MCMC simulasyonu) ile tahmin amaclanmistir.

### Yontem
Simulasyonda $\mu=150$ ve $\sigma=10$ degerleri yazilarak sentetik 50 adet veri (n=50) noktasindan olusan bir orneklem olusturulmustur. Yontem olarak **Bayesian Inference (Bayesyen Cikarim)** kullanilmistir. Python ekosistemindeki `emcee` (Markov Chain Monte Carlo) kutuphanesi ile posterior olasiligi hesaplanmis, 32 Walker ve 2000 adim kullanilarak gercege en yakin degere ulasilmak istenmistir. Uniform prior (0 < $\mu$ < 300, 0 < $\sigma$ < 50) ve Gaussyan olabilirlik (log-likelihood) modeli ile hedeflenen model egitilmistir.

### Sonuclar (Parametre Karsilastirma Tablosu)

| Degisken | Gercek Deger (Girdi) | Tahmin Edilen (Median) | Alt Sinir (%16) | Ust Sinir (%84) | Mutlak Hata |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $\mu$ (Parlaklik) | 150.0 | 147.78 | 146.39 | 149.12 | 2.22 |
| $\sigma$ (Hata Payi) | 10.0 | 9.47 | 8.63 | 10.58 | 0.53 |

### Bilimsel Yorumlar (Analiz Sorulari)

* **6.1. Merkezi Egilim ve Dogruluk (Accuracy) Analizi:**
Modelimizin tahmin ettigi ortalama deger ($\mu \approx 147.78$) gercek 150.0 degerine cok yakindir. Sadece 2.22 kadarlik ufak bir mutlak hata vererek, Bayesyen cikarimin isabet (accuracy) konusunda guclu oldugunu kanitlamistir.

* **6.2. Tahmin Hassasiyeti (Precision) Karsilastirmasi:**
Ortalamasi ($\mu$) alinan verinin varyansini ($\sigma$) tahmin etmek istatistiksel ve geometrik olarak daha zorludur cunku dagilimin ortasini (merkezini) bulmak sinirlari (sacilimi) tutturmaktan daha dar guven araligina sahiptir. Merkezi limit teoremi cercevesinde hata sapmasi, veri sayisi 50 de olsa $\mu$'de daha iyi saptanabilmektedir. Gozlem sayisini (n=50) daha da artirmadigimiz muddetce $\sigma$ parametresinin tahmin hassasiyeti daima daha az olacaktir. 

* **6.3. Olasiliksal Korelasyon Analizi (Corner Plot):**
Corner Plot uzerindeki dagilimin seklini veren elipsin buyuk oranda yatay ve dikey (eksenlere dik) olmasi $\mu$ ve $\sigma$ parametrelerinin birbirinden oldukca bagimsiz (zayif korelasyon/korelasyonsuz) hareket ettigini gostermektedir. Eger aralarinda guclu bir korelasyon (bagimlilik) olsaydi egik, kosegen bir elips grafigi gozlemler idik.

* **Prior Etkisi (Soru 1):**
Sayet $\mu$ icin 100-110 seklinde asiri dar ve hatali prior (on bilgi) verseydik, cikan Posterior dagilimi verinin agirligini gormezden gelip 110 sinirina birikerek yanli (bias) sonuc uretirdi, hakiki deger olan 150'yi asla yakalayamazdik.

* **Veri Miktari Etkisi (Soru 2):**
Gozlem sayisini (n=5) dusurdugumuzde veri kalitemiz cokecek, guven araligi (posterior belirsizligi) en az 3 katina kadar genisleyecekti. Sonuc daha guvensiz (low precision) ve genis aralikli olurdu.
