from hmmlearn import hmm
import numpy as np

# 'EV' kelimesi için model
model_ev = hmm.CategoricalHMM(n_components=2)
model_ev.startprob_ = np.array([1.0, 0.0])

# Geçiş ve emisyon matrislerini tanımlıyoruz (0 ve 1 sesleri ağırlıklı)
model_ev.transmat_ = np.array([[0.7, 0.3], 
                               [0.4, 0.6]])
model_ev.emissionprob_ = np.array([[0.8, 0.2, 0.0], 
                                   [0.1, 0.8, 0.1]])

# 'OKUL' kelimesi için model
model_okul = hmm.CategoricalHMM(n_components=2)
model_okul.startprob_ = np.array([1.0, 0.0])

# Geçiş ve emisyon matrisleri (2 sesi ağırlıklı)
model_okul.transmat_ = np.array([[0.6, 0.4], 
                                 [0.3, 0.7]])
model_okul.emissionprob_ = np.array([[0.1, 0.1, 0.8], 
                                     [0.0, 0.2, 0.8]])

def kelime_tani(test_verisi):
    # Hangi model daha yüksek puan veriyor?
    score_ev = model_ev.score(test_verisi)
    score_okul = model_okul.score(test_verisi)
    
    print(f"EV Modeli Puanı: {score_ev:.2f}")
    print(f"OKUL Modeli Puanı: {score_okul:.2f}")
    
    if score_ev > score_okul:
        print("Tahmin: EV")
    elif score_okul > score_ev:
        print("Tahmin: OKUL")
    else:
        print("Tahmin: Kararsız")
    print("-" * 25)

# Test verisi (Yeni bir ses kaydı geldiğini varsayalım)
print("1. Ses Kaydı Testi:")
test_data_1 = np.array([[0, 1, 1]]).T # Gözlemlerin indexleri
kelime_tani(test_data_1)

print("2. Ses Kaydı Testi:")
test_data_2 = np.array([[2, 2, 1]]).T
kelime_tani(test_data_2)