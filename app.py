import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "model_final.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

class_names = ["Blast", "Blight", "Tungro"]

st.title("Deteksi Penyakit Daun Padi Menggunakan CNN")

uploaded = st.file_uploader("Upload gambar daun...", type=["jpg","png","jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_resized = img.resize((224,224))
    img_array = np.expand_dims(np.array(img_resized)/255.0, axis=0)

    pred = model.predict(img_array)
    idx = np.argmax(pred)
    prob = np.max(pred)

    st.image(img, caption="Gambar Input", width=300)
    st.subheader("Hasil Prediksi:")
    st.write(f"Penyakit: **{class_names[idx]}**")
    st.write(f"Akurasi Prediksi: **{prob:.2f}**")
