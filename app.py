import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Klasifikasi Sayuran - MobileNetV2",
    page_icon="🥕",
    layout="centered"
)

IMG_SIZE = (224, 224)

# ===============================
# CHECK REQUIRED FILES
# ===============================
REQUIRED_FILES = [
    "model_mobilenetv2_5class.h5",
    "label_sayur_5class.json"
]

for file in REQUIRED_FILES:
    if not os.path.exists(file):
        st.error(f"File tidak ditemukan: {file}")
        st.stop()

# ===============================
# LOAD LABEL
# ===============================
with open("label_sayur_5class.json") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_mobilenetv2_5class.h5")
    return model

model = load_model()

# ===============================
# PREPROCESS IMAGE
# ===============================
def preprocess_image(img: Image.Image):
    img = img.resize(IMG_SIZE)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# ===============================
# UI
# ===============================
st.title("🥕 Klasifikasi Sayuran")
st.write("Model: **MobileNetV2 (Transfer Learning)**")

uploaded_file = st.file_uploader(
    "Upload gambar sayuran",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Input", width=300)

    img_array = preprocess_image(image)

    with st.spinner("Melakukan prediksi..."):
        prediction = model.predict(img_array)[0]

    idx = np.argmax(prediction)
    label = labels[idx]
    confidence = prediction[idx] * 100

    # ===============================
    # DISPLAY RESULT
    # ===============================
    st.success(f"🌱 Prediksi: **{label}**")
    st.metric("Confidence", f"{confidence:.2f}%")

    # ===============================
    # PROBABILITY CHART
    # ===============================
    st.markdown("### 📊 Probabilitas Tiap Kelas")
    prob_dict = {labels[i]: float(prediction[i]) for i in range(len(prediction))}
    st.bar_chart(prob_dict)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("Experiment Dashboard - MobileNetV2 | UAP Machine Learning")
