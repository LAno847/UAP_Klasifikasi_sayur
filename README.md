# 🥕Klasifikasi Sayuran Menggunakan Deep Learning

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
