import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Dashboard Klasifikasi Sayuran",
    page_icon="🥕",
    layout="wide"
)

IMG_SIZE = (224, 224)

# ===============================
# LOAD LABEL
# ===============================
with open("label_sayur_5class.json") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model("model_cnn_sayur.h5")
    mobilenet = tf.keras.models.load_model("model_mobilenetv2_5class.h5")
    vgg = tf.keras.models.load_model("model_vgg16_5class.h5")
    return cnn, mobilenet, vgg

cnn_model, mobilenet_model, vgg_model = load_models()

# ===============================
# PREPROCESS IMAGE
# ===============================
def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ===============================
# UI
# ===============================
st.title("🥕 Dashboard Klasifikasi Sayuran")
st.write("Prediksi menggunakan **CNN**, **MobileNetV2**, dan **VGG16**")

uploaded_file = st.file_uploader(
    "Upload gambar sayuran",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Input", width=300)

    img_array = preprocess_image(image)

    # ===============================
    # PREDICTION
    # ===============================
    cnn_pred = cnn_model.predict(img_array)[0]
    mobilenet_pred = mobilenet_model.predict(img_array)[0]
    vgg_pred = vgg_model.predict(img_array)[0]

    def get_result(pred):
        idx = np.argmax(pred)
        return labels[idx], float(pred[idx]) * 100

    cnn_label, cnn_conf = get_result(cnn_pred)
    mob_label, mob_conf = get_result(mobilenet_pred)
    vgg_label, vgg_conf = get_result(vgg_pred)

    # ===============================
    # DISPLAY RESULTS
    # ===============================
    st.markdown("## 📊 Hasil Prediksi")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🧠 CNN", cnn_label, f"{cnn_conf:.2f}%")

    with col2:
        st.metric("⚡ MobileNetV2", mob_label, f"{mob_conf:.2f}%")

    with col3:
        st.metric("🏗️ VGG16", vgg_label, f"{vgg_conf:.2f}%")

    # ===============================
    # DETAIL PROBABILITIES
    # ===============================
    st.markdown("## 🔍 Detail Probabilitas")

    st.write("### CNN")
    st.bar_chart(cnn_pred)

    st.write("### MobileNetV2")
    st.bar_chart(mobilenet_pred)

    st.write("### VGG16")
    st.bar_chart(vgg_pred)
