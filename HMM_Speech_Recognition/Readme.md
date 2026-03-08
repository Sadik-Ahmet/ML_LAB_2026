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

Analiz ve Sonuçlar 

### Gürültü Etkisi

Ses verisindeki gürültü, HMM modelindeki **Emisyon Olasılıklarını (B matrisi)** doğrudan etkiler. Gürültü, gözlem dizisindeki (High/Low) netliği bozarak fonemler arasındaki ayırt ediciliği azaltır ve modelin hata payını artırır.

### Neden Derin Öğrenme?

Gerçek dünyadaki binlerce kelimelik sistemlerde Viterbi ve geleneksel HMM yerine Derin Öğrenme (Deep Learning) tercih edilmektedir çünkü: 

* Derin öğrenme modelleri (LSTM, Transformer), uzun vadeli bağlamsal ilişkileri daha iyi yakalar.
* Büyük veri setlerinde ölçeklenebilirlik ve doğruluk oranları HMM'den çok daha yüksektir.
* Manuel özellik çıkarımı (feature engineering) yerine uçtan uca öğrenme imkanı sunar.


## Teslim Bilgileri

* **Öğrenci:** Sadık Ahmet Çağlın
* **Numara:** 23291386
