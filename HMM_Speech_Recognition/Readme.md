Proje Yapısı 

Ödev isterlerine uygun olarak oluşturulan dizin yapısı şu şekildedir:

```
HMM-Speech-Recognition/
├── src/            
   └── recognizer.py  # HMM modellerini ve sınıflandırma fonksiyonunu içeren kod [cite: 48]
├── report/         
   └── cozum_anahtari.pdf  # Viterbi el hesaplamaları ve teorik analizler [cite: 50]
├── requirements.txt # Gerekli kütüphaneler (hmmlearn, numpy) [cite: 51]
└── README.md       # Proje açıklaması ve dokümantasyon [cite: 52]

```
---

# HMM ile İzole Kelime Tanıma (Speech Recognition)

Bu proje, Gizli Markov Modelleri (HMM) kullanarak "EV" ve "OKUL" gibi izole kelimelerin tanınmasını sağlayan bir sistemin teorik ve pratik uygulamasını içerir.

## 1. Problem Tanımı

Konuşma sinyalleri, fonem adı verilen ses birimlerinden oluşur. Temel problem, mikrofondan gelen belirsiz ve gürültülü gözlem dizilerini (ses spektrumu) analiz ederek, bu dizinin hangi kelimeye ("EV" mi "OKUL" mu) ait olduğunu en yüksek olasılıkla tespit etmektir.

## 2. Veri

* 
**Gizli Durumlar ($S$):** Kelimeyi oluşturan fonemler (Örn: "e", "v").


* 
**Gözlemler ($O$):** Sesin frekans karakteristiğini temsil eden "High" ve "Low" değerleri.


* 
**Eğitim Verisi:** Her kelime için önceden tanımlanmış gözlem dizileri.



## 3. Yöntem

* 
**Teorik Hesaplama:** En olası fonem dizisini bulmak için **Viterbi Algoritması** kullanılmıştır.


* 
**Uygulama:** Python'da `hmmlearn` kütüphanesi kullanılarak her kelime için ayrı HMM modelleri tanımlanmıştır.


* 
**Sınıflandırma:** Yeni gelen bir ses verisinin hangi modelde daha yüksek **Log-Likelihood** (olasılık puanı) verdiği hesaplanmıştır.



## 4. Sonuçlar

* Viterbi algoritması ile adım adım yapılan hesaplamalar sonucunda, belirli bir gözlem dizisi (Örn: [High, Low]) için en olası fonem yolu (Örn: "e-v") başarıyla tespit edilmiştir.


* Python simülasyonunda, test verileri ilgili modelde (EV veya OKUL) en yüksek skoru alarak doğru sınıflandırılmıştır.



## 5. Yorum ve Tartışma

* 
**Gürültü Etkisi:** Ses verisindeki gürültü, Emisyon Olasılıklarını ($B$ matrisi) bozarak gözlemlerin ayırt ediciliğini düşürür ve yanlış sınıflandırmalara neden olur.


* 
**DL vs. HMM:** Binlerce kelimelik gerçek sistemlerde, HMM'in sınırlı bağlam kapasitesi ve yüksek hesaplama maliyeti nedeniyle, uzun vadeli ilişkileri daha iyi kurabilen Derin Öğrenme yapıları tercih edilmektedir.



---

## Teslim Bilgileri

* **Öğrenci:** Sadık Ahmet Çağlın
* **Numara:** 23291386
