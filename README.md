# 🥕Klasifikasi Sayuran Menggunakan Deep Learning

<p align="center">
  <b>Deep Learning-Based Vegetable Image Classification using CNN, MobileNetV2, and VGG</b><br>
  Streamlit Web Dashboard for Real-Time Image Prediction
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange"/>
  <img src="https://img.shields.io/badge/Web%20App-Streamlit-red"/>
  <img src="https://img.shields.io/badge/Image%20Classification-CNN-blue"/>
</p>

## 📌 Deskripsi Proyek
Proyek ini bertujuan untuk membangun sistem **klasifikasi gambar sayuran** berbasis **Deep Learning** yang mampu mengenali jenis sayuran dari sebuah gambar. Sistem ini mengimplementasikan dan membandingkan **tiga model CNN**, yaitu:
1. CNN from Scratch
2. MobileNetV2 (Transfer Learning)
3. VGG16 (Transfer Learning)

Hasil prediksi dari ketiga model ditampilkan dalam sebuah **dashboard berbasis web menggunakan Streamlit**, sehingga pengguna dapat mengunggah gambar sayuran dan melihat hasil klasifikasi beserta tingkat kepercayaannya (confidence).

Proyek ini dikembangkan sebagai bagian dari tugas **Ujian Akhir Praktikum / Praktikum Machine Learning**.

---

## 📂 Dataset dan Preprocessing

### 🔹 Dataset
Dataset yang digunakan merupakan dataset citra sayuran yang disusun ke dalam **5 kelas**, yaitu:
- **Carrot**
- **Cucumber**
- **Potato**
- **Radish**
- **Tomato**

Struktur dataset:

Vegetable Images/
├── train/
│ ├── Carrot/
│ ├── Cucumber/
│ ├── Potato/
│ ├── Radish/
│ └── Tomato/
├── val/
└── test/



Dataset dibagi menjadi tiga bagian:
- **Train**: untuk pelatihan model
- **Validation**: untuk validasi selama training
- **Test**: untuk evaluasi akhir model

---

### 🔹 Preprocessing Data
Tahapan preprocessing yang dilakukan:
- Resize gambar menjadi **224 × 224**
- Normalisasi pixel dengan skala **1/255**
- Data augmentation (khusus data train):
  - Rotasi
  - Zoom
  - Horizontal flip

Preprocessing dilakukan menggunakan `ImageDataGenerator` dari TensorFlow/Keras.

---

## 🧠 Model yang Digunakan

### 1️⃣ CNN from Scratch
Model CNN dasar dibangun tanpa pretrained weights.
- Beberapa layer **Conv2D + MaxPooling**
- Fully Connected Layer
- Dropout untuk mengurangi overfitting
- Cocok sebagai baseline model

**Kelebihan:**
- Arsitektur sederhana
- Mudah dipahami

**Kekurangan:**
- Akurasi relatif lebih rendah
- Training lebih lambat untuk performa optimal

---

### 2️⃣ MobileNetV2 (Transfer Learning)
MobileNetV2 menggunakan pretrained weights dari **ImageNet**.
- Base model di-freeze
- Ditambahkan Global Average Pooling dan Dense Layer
- Ringan dan efisien

**Kelebihan:**
- Model ringan
- Cepat saat inferensi
- Cocok untuk aplikasi real-time

**Kekurangan:**
- Sedikit lebih sensitif terhadap kualitas data

---

### 3️⃣ VGG16 (Transfer Learning)
VGG16 juga menggunakan pretrained weights dari **ImageNet**.
- Arsitektur deep dan stabil
- Base model di-freeze
- Custom classification head

**Kelebihan:**
- Akurasi stabil
- Feature extraction kuat

**Kekurangan:**
- Ukuran model besar
- Waktu inferensi lebih lama

---

## 📊 Hasil Evaluasi dan Analisis Perbandingan

Evaluasi dilakukan menggunakan **data test** dengan metrik:
- Accuracy

### 🔹 Ringkasan Perbandingan Model

| Model         | Akurasi | Kecepatan Inferensi | Ukuran Model |
|---------------|---------|---------------------|--------------|
| CNN           | Tinggi  | Sedang              | Kecil        |
| MobileNetV2   | Tinggi  | Sangat Cepat        | Sangat Kecil |
| VGG16         | Tinggi  | Lebih Lambat        | Besar        |

### 🔹 Analisis
- **CNN** cocok sebagai baseline, namun performanya masih kalah dibanding transfer learning.
- **MobileNetV2** memberikan keseimbangan terbaik antara akurasi dan kecepatan.
- **VGG16** memberikan hasil stabil tetapi kurang efisien untuk deployment ringan.

Berdasarkan hasil tersebut, **MobileNetV2** merupakan model paling optimal untuk sistem klasifikasi sayuran ini.

---

## 🌐 Panduan Menjalankan Website (Dashboard) Secara Lokal

### 🔹 1. Persiapan Lingkungan
Pastikan Python sudah terinstall (disarankan Python ≥ 3.9).

Install dependency:
```bash
pip install streamlit tensorflow pillow numpy

