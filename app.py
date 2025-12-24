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
    mobilenet = tf.keras.models.load_model("model_mobilenetv2_5class.h5")
    
    return mobilenet

 mobilenet_model = load_models()

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
    
    mobilenet_pred = mobilenet_model.predict(img_array)[0]
   

    def get_result(pred):
        idx = np.argmax(pred)
        return labels[idx], float(pred[idx]) * 100

   
    mob_label, mob_conf = get_result(mobilenet_pred)
    

    # ===============================
    # DISPLAY RESULTS
    # ===============================
    st.markdown("## 📊 Hasil Prediksi")

    col1 = st.columns(3)

   

    with col1:
        st.metric("⚡ MobileNetV2", mob_label, f"{mob_conf:.2f}%")

    

    # ===============================
    # DETAIL PROBABILITIES
    # ===============================
    st.markdown("## 🔍 Detail Probabilitas")

   

    st.write("### MobileNetV2")
    st.bar_chart(mobilenet_pred)

   


