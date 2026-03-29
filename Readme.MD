# YZM212 Makine Öğrenmesi – III. Laboratuvar Değerlendirmesi

**Tarih:** 29.03.2026  
**Konu:** Matris Manipülasyonu, Özdeğerler ve Özvektörler  

---

## İçindekiler
1. [Matris Manipülasyonu, Özdeğerler ve Özvektörlerin Makine Öğrenmesi ile İlişkisi](#soru-1)  
2. [NumPy `linalg.eig` Fonksiyonu: Dokümantasyon ve Kaynak Kod İncelemesi](#soru-2)  
3. [Sıfırdan Özdeğer Hesaplama ve NumPy ile Karşılaştırma](#soru-3)  
4. [Kaynaklar](#kaynaklar)

---

## Soru 1 – Matris Manipülasyonu, Özdeğerler ve Özvektörlerin Makine Öğrenmesi ile İlişkisi <a name="soru-1"></a>

### Temel Tanımlar

**Matris**, bir doğrusal dönüşümü (lineer transformation) sayısal olarak temsil eden dikdörtgen bir sayı dizisidir. Boyutlar $m \times n$ (satır × sütun) şeklinde gösterilir. Makine öğrenmesinde veriler genellikle matrisler olarak temsil edilir: $n$ gözlem ve $p$ özellik içeren bir veri kümesi $\mathbf{X} \in \mathbb{R}^{n \times p}$ matrisi olarak ifade edilir.

**Matris Manipülasyonu**, bu matrisler üzerinde gerçekleştirilen işlemler bütünüdür:  

| İşlem | Açıklama |
|---|---|
| Toplama / Çıkarma | Aynı boyutlu matrislerin eleman bazlı işlemi |
| Matris Çarpımı | $\mathbf{C} = \mathbf{A}\mathbf{B}$; ara boyutların eşit olması gerekir |
| Transpoze | Satır ve sütunların yer değiştirilmesi, $\mathbf{A}^T$ |
| Tersine Matris | $\mathbf{A}^{-1}$ ile $\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$ sağlanır |
| Determinant | Bir matrisin skaler ölçeği; sıfır ise matris tekil (singular) |
| Norm | Matrisin "büyüklüğünü" ölçen skaler değer (L1, L2, Frobenius vb.) |

---

**Özdeğer (Eigenvalue)** ve **Özvektör (Eigenvector)**, bir kare matrisin $\mathbf{A}$ temsil ettiği doğrusal dönüşümün *yönünü koruyarak yalnızca uzatan ya da daraltan* özel vektörler ve ilgili ölçeklerdir. Matematiksel tanım:

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

- $\mathbf{v} \neq \mathbf{0}$ : özvektör (eigenvector)  
- $\lambda$ : özdeğer (eigenvalue); gerçek ya da karmaşık sayı olabilir  

Özvektörler, $\mathbf{A}$'nın etkisinin *yön değiştirmeksizin sadece ölçeklendiği* doğrultulardır. Özdeğer, o ölçekleme miktarını verir.

**Özdeğer hesaplama:** Karakteristik denklem çözülerek bulunur:

$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$$

$n \times n$ bir matris için en fazla $n$ özdeğer vardır.

---

### Makine Öğrenmesindeki Kullanım Alanları

#### 1. Temel Bileşen Analizi (Principal Component Analysis – PCA)

PCA, yüksek boyutlu verileri düşük boyutlu bir uzaya *bilgi kaybını minimize ederek* yansıtır. Adımlar:

1. Veriyi ortalama merkezle: $\tilde{\mathbf{X}} = \mathbf{X} - \bar{\mathbf{X}}$  
2. Kovaryans matrisini hesapla: $\mathbf{C} = \frac{1}{n-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}}$  
3. $\mathbf{C}$'nin özdeğer-özvektörlerini bul: $\mathbf{C}\mathbf{v}_i = \lambda_i \mathbf{v}_i$  
4. Özdeğerleri büyükten küçüğe sırala; en büyük $k$ özdeğere karşılık gelen özvektörler (temel bileşenler) seçilir  
5. Veriyi bu özvektörlerin uzayına yansıt: $\mathbf{Z} = \tilde{\mathbf{X}}\mathbf{V}_k$  

Büyük özdeğer → o yönde yüksek varyans → daha fazla bilgi. PCA; boyut indirgeme, görselleştirme, gürültü azaltma ve overfitting önlemede yaygın kullanılır.

#### 2. Tekil Değer Ayrışımı (Singular Value Decomposition – SVD)

$$\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$$

Dikdörtgeni de kapsayan SVD, $\mathbf{X}^T\mathbf{X}$ ve $\mathbf{X}\mathbf{X}^T$'nin özdeğerleriyle ilişkilidir. Tekil değerler ($\sigma_i = \sqrt{\lambda_i}$), matrisin "enerji dağılımını" gösterir. Kullanım alanları: tavsiye sistemleri (Netflix), doğal dil işleme (LSA), görüntü sıkıştırma.

#### 3. Lineer Diskriminant Analizi (LDA)

Sınıfları en iyi ayıran doğrultular, sınıf-içi ve sınıf-arası saçılım matrislerinin genelleştirilmiş özdeğer problemi çözülerek bulunur:

$$\mathbf{S}_W^{-1}\mathbf{S}_B \mathbf{v} = \lambda\mathbf{v}$$

#### 4. Spektral Kümeleme (Spectral Clustering)

Grafın Laplace matrisinin küçük özdeğerlerine karşılık gelen özvektörler, veriyi kümelere ayırmak için kullanılır. Karmaşık topological yapıları yakalamada K-Means'e göre üstündür.

#### 5. Destek Vektörü Makineleri (SVM) – Kernel Matrisi

RBF gibi çekirdek fonksiyonlarıyla oluşturulan Gram matrisinin özdeğerleri, çekirdek matrisinin pozitif yarı-tanımlı olduğunu doğrulamak için kontrol edilir.

#### 6. Graf Sinir Ağları (GNN) ve Spektral Yöntemler

Grafın normalize Laplace matrisinin özdeğer-özvektörleri, spektral evrişim tabanlı GNN'lerde doğrudan kullanılır.

#### 7. Matris Ayrışımı ile Gizli Anlam Analizi

Doküman-terim matrisinin SVD ile ayrıştırılması, tür/kategori yapısını açığa çıkarır.

#### 8. Covariance-Based Regularization

Ridge regresyonunda $(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}$ matrisinin kondisyon sayısı (en büyük / en küçük özdeğer oranı), sayısal kararlılığı ölçer.

---

## Soru 2 – NumPy `linalg.eig` Fonksiyonu: Dokümantasyon ve Kaynak Kod İncelemesi <a name="soru-2"></a>

### 2.1 Dokümantasyon Özeti

**Kaynak:** [numpy.linalg.eig — NumPy v2.1 Manual](https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html)

```python
numpy.linalg.eig(a)
```

| Parametre | Açıklama |
|---|---|
| `a` | `(..., M, M)` boyutunda kare matrisler; son iki eksen boyunca EigenAyrışımı yapılır |

**Dönen Değerler:**

| Değer | Açıklama |
|---|---|
| `eigenvalues` | `(..., M)` boyutunda dizi; her matrisin $M$ özdeğerini içerir. Karmaşık tip; sanal kısım sıfırsa gerçek tipe dönüşür |
| `eigenvectors` | `(..., M, M)` boyutunda dizi; `eigenvectors[:,i]`, `eigenvalues[i]`'ye karşılık gelen *sağ* özvektördür (normalize edilmiş, birim uzunluklu) |

**Matematiksel Tanım:**  
$$\mathbf{A} \cdot \mathbf{v}[:, i] = \lambda_i \cdot \mathbf{v}[:, i]$$

**İstisna:** Özdeğer hesabı yakınsamazsa `LinAlgError` fırlatılır.

**Önemli Notlar:**
- Özdeğerler sıralı döndürülmez (büyükten küçüğe ya da sıralı değil)
- Özvektörler *sağ* özvektörlerdir; sol özvektör için transpoze + ters matris gerekmez
- Simetrik/Hermitian matrisler için `eigh` tercih edilmeli (daha hızlı ve sayısal olarak daha kararlı)
- Geniş yayın (broadcasting) kuralları desteklenir  

**Kullanım Örneği:**

```python
import numpy as np
from numpy import linalg as LA

A = np.diag([1, 2, 3])
eigenvalues, eigenvectors = LA.eig(A)
# eigenvalues → array([1., 2., 3.])
# eigenvectors → array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
```

---

### 2.2 Kaynak Kod İncelemesi

**Kaynak:** [`numpy/linalg/_linalg.py`, L1345-L1499](https://github.com/numpy/numpy/blob/v2.1.0/numpy/linalg/_linalg.py#L1345-L1499)

NumPy'ın `eig` fonksiyonu aşağıdaki temel adımlardan oluşmaktadır:

#### Adım 1 – Girdi Doğrulama

```python
a, wrap = _makearray(a)
_assert_stacked_2d(a)
_assert_stacked_square(a)
t, result_t = _commonType(a)
```

- `_makearray`: Girdi diziye dönüştürülür (gerekirse kopyalanır).
- `_assert_stacked_2d` / `_assert_stacked_square`: Boyut kontrolleri yapılır; matrisin en az 2D ve kare olduğu garanti edilir.
- `_commonType`: Girdi `float64` ya da `complex128`'e yükseltilir.

#### Adım 2 – LAPACK Rutini Seçimi

```python
if isComplexType(t):
    lapack_routine = lapack_lite.zgeev
else:
    lapack_routine = lapack_lite.dgeev
```

- Matris tipi gerçek ise `dgeev` (double precision), karmaşık ise `zgeev` (complex double) LAPACK rutini seçilir.
- Bu LAPACK rutinleri (Fortran tabanlı), dünya genelindeki lineer cebir kütüphanelerinin temelini oluşturur.

#### Adım 3 – LAPACK Çağrısı

```python
results = _umath_linalg.eig(a, signature=signature)
```

- Yayın aritmetiği (broadcasting) ile optimize edilmiş C uzantısı üzerinden LAPACK çağrılır.
- LAPACK `_geev` ailesi, **QR algoritması** ile Hessenberg formadan eigenvalueları hesaplar.

#### Adım 4 – Sonuç İşleme

```python
w = results[0]  # Özdeğerler
vt = results[1] # Özvektörler (transpose)
```

- Gerçek `dgeev` rutini, karmaşık özdeğerleri (conjugate çiftler) ayırarak döndürür; NumPy bunları birleştirir.
- Sanal kısım sıfırsa `real` tipine dönüşüm uygulanır.

#### Özetlenen Akış

```
Girdi Matrisi A (M×M)
       │
       ▼
 Tip & Boyut Doğrulama
       │
       ▼
 LAPACK Rutini Seçimi (dgeev / zgeev)
       │
       ▼
 Hessenberg Dönüşümü (QR iterasyonuna hazırlık)
       │
       ▼
 QR Algoritması ile Schur Ayrışımı
       │
       ▼
 Geri Dönüşüm: Özdeğerler (w) + Özvektörler (v)
       │
       ▼
 Tip Normalizasyonu & Broadcasting
       │
       ▼
 eigenvalues: (..., M)
 eigenvectors: (..., M, M)
```

**Neden LAPACK?**  
LAPACK (Linear Algebra PACKage), sayısal kararlılık ve performans açısından 50+ yıl boyunca optimize edilmiş Fortran kodudur. `_geev` (General EigenValue) rutini özellikle genel kare matrisler için tasarlanmış olup QR iterasyonunun öncesinde Hessenberg indirgeme (Householder yansımaları) uygular. Bu, $O(n^3)$ karmaşıklıkla verimli bir hesaplama sağlar.

---

## Soru 3 – Sıfırdan Özdeğer Hesaplama ve NumPy ile Karşılaştırma <a name="soru-3"></a>

**Referans:** [LucasBN/Eigenvalues-and-Eigenvectors](https://github.com/LucasBN/Eigenvalues-and-Eigenvectors)

> Detaylı uygulama ve kod için bkz: [`EigenVectorsValues.ipynb`](./EigenVectorsValues.ipynb)

### 3.1 Kullanılan Yöntem: Kuvvet İterasyonu + Hotelling Deflasyonu

**Kuvvet İterasyonu (Power Iteration):**  
Dönüşüm matrisinin defalarca uygulanmasıyla vektörün dominant özvektöre yakınsadığını gösteren iteratif bir yöntemdir.

$$\mathbf{b}_{k+1} = \frac{\mathbf{A}\mathbf{b}_k}{\|\mathbf{A}\mathbf{b}_k\|}$$

Yakınsama sonrası özdeğer, **Rayleigh Quotient** ile elde edilir:

$$\lambda = \frac{\mathbf{b}^T \mathbf{A} \mathbf{b}}{\mathbf{b}^T \mathbf{b}}$$

**Hotelling Deflasyonu:**  
Bir özdeğer-özvektör çifti ($\lambda_1$, $\mathbf{v}_1$) bulunduktan sonra matris deflate edilerek bir sonraki dominant özdeğer aranır:

$$\mathbf{A}' = \mathbf{A} - \lambda_1 \mathbf{v}_1 \mathbf{v}_1^T$$

Bu işlem $n$ kez tekrarlanarak tüm özdeğerler elde edilir.

### 3.2 Test Matrisi

```
A = [[4, 2, 0],
     [2, 3, 1],
     [0, 1, 2]]
```

Simetrik matris seçildi → gerçek özdeğerler garanti edilir.

### 3.3 Karşılaştırma Sonuçları

| Özdeğer | Custom (Power Iteration) | NumPy `eig` | Fark (abs) |
|:---:|:---:|:---:|:---:|
| λ₁ | 5.669079 | 5.669079 | < 1e-9 |
| λ₂ | 2.476024 | 2.476024 | < 1e-9 |
| λ₃ | 0.854897 | 0.854897 | < 1e-9 |

Her iki yöntem de aynı özdeğerleri bulmaktadır. Küçük farklar sayısal hassasiyet ve iterasyon toleransından kaynaklanmaktadır.

### 3.4 Karşılaştırmalı Değerlendirme

| Özellik | Kuvvet İterasyonu | NumPy `eig` |
|---|---|---|
| **Yöntem** | İteratif (yaklaşık) | LAPACK QR (tam) |
| **Karmaşıklık** | $O(n^2 \cdot k)$ ($k$ = iterasyon) | $O(n^3)$ |
| **Hassasiyet** | Toleransa bağlı | Makine hassasiyeti |
| **Karmaşık özdeğerler** | Kısıtlı | Tam destek |
| **Kullanım amacı** | Eğitim / büyük seyrek matrisler | Genel amaçlı |
| **NumPy bağımlılığı** | Yalnızca temel işlemler | LAPACK |

**Sonuç:** NumPy'ın `eig` fonksiyonu üretim ortamlarında ve genel kullanımda tercih edilmelidir. Kuvvet iterasyonu ise algoritmayı anlamak, büyük seyrek matrislerde yalnızca birkaç özdeğer istendiğinde (PageRank, spektral kümeleme gibi) veya eğitim amaçlı kullanılmak için değerlidir.

---

## Kaynaklar <a name="kaynaklar"></a>

1. **Jason Brownlee** – *Introduction to Matrices and Matrix Arithmetic for Machine Learning*  
   [MachineLearningMastery.com](https://machinelearningmastery.com/introduction-matrices-machine-learning/)

2. **Jason Brownlee** – *Gentle Introduction to Eigenvalues and Eigenvectors for Machine Learning*  
   [MachineLearningMastery.com](https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/)

3. **NumPy Documentation** – *numpy.linalg.eig*  
   [numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html](https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html)

4. **NumPy Source Code** – *numpy/linalg*  
   [github.com/numpy/numpy/tree/main/numpy/linalg](https://github.com/numpy/numpy/tree/main/numpy/linalg)

5. **LucasBN** – *Eigenvalues and Eigenvectors (Custom Implementation)*  
   [github.com/LucasBN/Eigenvalues-and-Eigenvectors](https://github.com/LucasBN/Eigenvalues-and-Eigenvectors)

6. **G. Strang** – *Linear Algebra and Its Applications*, 2nd Ed., Academic Press, 1980.

7. **Golub & Van Loan** – *Matrix Computations*, 4th Ed., Johns Hopkins University Press, 2013.

8. **LAPACK Users' Guide** – *dgeev/zgeev routines*  
   [netlib.org/lapack/lug](https://www.netlib.org/lapack/lug/)

---

*Bu rapor YZM212 Makine Öğrenmesi dersi III. Laboratuvar değerlendirmesi kapsamında hazırlanmıştır.*
