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
Dataset yang digunakan adalah **Vegetable Image Dataset** yang diperoleh dari Kaggle:

- **Link**: [Kaggle - Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)

Dataset ini berisi kumpulan citra berbagai jenis sayuran yang digunakan untuk keperluan klasifikasi citra berbasis deep learning.

Pada proyek ini, digunakan **5 kelas sayuran**, yaitu:
- **Carrot**
- **Cucumber**
- **Potato**
- **Radish**
- **Tomato**

Dataset dibagi ke dalam tiga bagian:
- **Training set**: data untuk melatih model
- **Validation set**: data untuk memantau performa selama training
- **Testing set**: data untuk evaluasi akhir model


Struktur dataset:


| train         | val       | test     |
|---------------|-----------|----------|
| Carrot        | Carrot    |Carrot    |
| Cucumber      | Cucumber  |Cucumber  |
| Potato        | Potato    |Potato    |
| Radish        | Radish    |Radish    |
| Tomato        | Tomato    |Tomato    |


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

---

### 2️⃣ MobileNetV2 (Transfer Learning)
MobileNetV2 menggunakan pretrained weights dari **ImageNet**.
- Base model di-freeze
- Ditambahkan Global Average Pooling dan Dense Layer
- Ringan dan efisien

---

### 3️⃣ VGG16 (Transfer Learning)
VGG16 juga menggunakan pretrained weights dari **ImageNet**.
- Arsitektur deep dan stabil
- Base model di-freeze
- Custom classification head

---

## 📊 Hasil Evaluasi dan Analisis Perbandingan

Evaluasi dilakukan menggunakan **data test** dengan metrik:
- Accuracy

### 🔹 Ringkasan Perbandingan Model

| Model         | Akurasi | Kecepatan Inferensi | Ukuran Model |
|---------------|---------|---------------------|--------------|
| CNN           | 96%     | Sedang              | Kecil        |
| MobileNetV2   | 99%     | Sangat Cepat        | Sangat Kecil |
| VGG16         | 99%     | Lebih Lambat        | Besar        |

### 🔹 Analisis
- **CNN** cocok sebagai baseline, namun performanya masih kalah dibanding transfer learning.
- **MobileNetV2** memberikan keseimbangan terbaik antara akurasi dan kecepatan.
- **VGG16** memberikan hasil stabil tetapi kurang efisien untuk deployment ringan.

Berdasarkan hasil tersebut, **MobileNetV2** merupakan model paling optimal untuk sistem klasifikasi sayuran ini.

---

## 🌐 Dashboard Klasifikasi Sayuran
Dashboard dikembangkan menggunakan **Streamlit** dengan fitur:
- Upload gambar sayuran
- Prediksi menggunakan model Deep Learning
- Menampilkan label kelas dan confidence
- Visualisasi probabilitas prediksi

---

## ▶️ Panduan Menjalankan Sistem Secara Lokal

### 🔹 1. Persiapan Lingkungan
Pastikan Python telah terinstall (disarankan Python ≥ 3.9).

Install dependency:
```bash
pip install streamlit tensorflow numpy pillow
```

📁 Project Structure
```
UAP_Klasifikasi_sayur/
├── app.py
├── label_sayur_5class.json
├── model_cnn_sayur.h5
├── model_mobilenetv2_5class.h5
├── model_vgg16_5class.h5

```
▶️ Menjalankan Dashboard
```
-streamlit run app.py
```

🌐 Buka di browser:
```
-http://localhost:8501
```

---

### Link Live Demo

Coba aplikasi Klasifikasi Sayuran Menggunakan Deep Learning secara langsung dengan mengunjungi tautan di bawah ini:

[🔗 **Demo Aplikasi Sederhana Streamlit**](https://uapklasifikasisayur-86pf48snctnwy659mfmbjz.streamlit.app/)


---

<h1 align="center">👤 Biodata 👤</h1>

👤 **[Bastian Feraries Wijaya](https://github.com/LAno847)**  
📘 **NIM**: 202210370311323
🎓 **Program Studi**: Teknik Informatika  
🏛️ **Universitas Muhammadiyah Malang**

---
