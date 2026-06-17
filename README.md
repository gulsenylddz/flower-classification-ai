<div align="center">

# 🌸 Flora AI

### Yapay Zeka Destekli Çiçek Türü Tanıma ve Polen Risk Analizi Sistemi

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![MobileNetV2](https://img.shields.io/badge/CNN-MobileNetV2-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

</div>

---

## 📌 Proje Hakkında

Flora AI, görüntü işleme ve derin öğrenme teknikleri kullanılarak geliştirilmiş bir çiçek sınıflandırma sistemidir.

Sistem, kullanıcı tarafından yüklenen çiçek görüntüsünü analiz ederek:

✅ Çiçek türünü belirler

✅ Tahmin güven skorunu hesaplar

✅ Polen risk seviyesini gösterir

✅ Grad-CAM ile modelin karar mekanizmasını görselleştirir

---

## 🎯 Projenin Amacı

Bu proje, derin öğrenme tabanlı görüntü sınıflandırma yöntemleri kullanılarak çiçek türlerinin otomatik olarak tanınmasını amaçlamaktadır.

Geliştirilen sistem;

- Görüntü ön işleme tekniklerini
- CNN mimarilerini
- Transfer Learning yaklaşımını
- Açıklanabilir Yapay Zeka (Grad-CAM)

tek bir uygulama içerisinde birleştirmektedir.

---

# 🖼️ Veri Seti

| Özellik | Değer |
|----------|----------|
| Veri Seti | Flower Classification |
| Sınıf Sayısı | 14 |
| Eğitim Görüntüsü | 13.642 |
| Doğrulama Görüntüsü | 98 |
| Görüntü Boyutu | 224x224 |
| Format | JPG / PNG |

### Kullanılan Çiçek Türleri

🌷 Tulip

🌹 Rose

🌻 Sunflower

🌼 Common Daisy

🌸 Carnation

🌺 Iris

💐 Astilbe

ve diğer toplam 14 farklı çiçek türü.

---

# ⚙️ Görüntü İşleme Pipeline'ı

```text
Görüntü Yükleme
        ↓
Resize (224x224)
        ↓
Normalization
        ↓
Data Augmentation
        ↓
Tensor Dönüşümü
        ↓
MobileNetV2
        ↓
Sınıflandırma
        ↓
Polen Risk Analizi
        ↓
Grad-CAM Görselleştirme
```

---

# 🧠 Kullanılan Yapay Zeka Modeli

## MobileNetV2

Projede CNN tabanlı MobileNetV2 mimarisi kullanılmıştır.

### Tercih Edilme Sebepleri

- Yüksek doğruluk
- Düşük hesaplama maliyeti
- Hızlı eğitim süreci
- Mobil sistemlere uygunluk
- Transfer Learning desteği

---

# 🔄 Transfer Learning

Sıfırdan model eğitmek yerine önceden eğitilmiş MobileNetV2 modeli kullanılmıştır.

Bu sayede:

✔ Eğitim süresi azalmıştır

✔ Daha yüksek doğruluk elde edilmiştir

✔ Veri seti daha verimli kullanılmıştır

---

# 📈 Eğitim Sonuçları

| Metrik | Sonuç |
|----------|----------|
| Eğitim Doğruluğu | %90.2 |
| Doğrulama Doğruluğu | %85-89 |
| Sınıf Sayısı | 14 |
| Giriş Boyutu | 224x224 |

---

# 🔥 Grad-CAM

Grad-CAM yöntemi kullanılarak modelin tahmin sırasında görüntünün hangi bölgelerine odaklandığı analiz edilmiştir.

Bu özellik sayesinde sistem yalnızca tahmin yapmakla kalmaz, aynı zamanda kararının nedenini de görselleştirir.

---

# 🛠️ Kullanılan Teknolojiler

- Python
- TensorFlow
- Keras
- MobileNetV2
- Streamlit
- OpenCV
- NumPy
- Pillow

---

# 🚀 Kurulum

```bash
git clone https://github.com/kullaniciadi/flora-ai.git

cd flora-ai

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

---

# 📂 Proje Yapısı

```text
flora-ai
│
├── dataset
├── model
│   └── flower_model.h5
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

# 🔮 Gelecek Çalışmalar

- 📱 Mobil uygulama geliştirme
- 📷 Canlı kamera desteği
- 🌍 Daha büyük veri setleri
- 🤖 Gelişmiş polen tahmin sistemi
- ☁️ Bulut tabanlı dağıtım

---

<div align="center">

### 👨‍💻 Geliştirici

## Mertcan Topcu

Bilgisayar Mühendisliği

⭐ Projeyi beğendiysen yıldız vermeyi unutma ⭐

</div>