import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =====================================
# 1. Load Model
# =====================================
MODEL_PATH = "model_B1.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Nama kelas sesuai dataset
class_names = ["Blast", "Blight", "Tungro"]

# =====================================
# 2. Tampilan Aplikasi
# =====================================
st.title("🌾 Deteksi Penyakit Daun Padi Menggunakan CNN")
st.write("Upload gambar daun padi untuk mendeteksi jenis penyakitnya.")

uploaded = st.file_uploader("Upload gambar daun...", type=["jpg", "png", "jpeg"])

# =====================================
# 3. Proses Prediksi
# =====================================
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    # --- BACA UKURAN INPUT MODEL SECARA OTOMATIS ---
    input_shape = model.input_shape  # contoh: (None, 150,150,3)
    img_w = input_shape[1]
    img_h = input_shape[2]

    st.write(f"📌 Ukuran input model: {img_w} x {img_h}")

    img_resized = img.resize((img_w, img_h))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    # Prediksi
    pred = model.predict(img_array)
    idx = np.argmax(pred)
    prob = np.max(pred)

    # =====================================
    # 4. Output Hasil Prediksi
    # =====================================
    st.image(img, caption="Gambar yang diupload", use_column_width=True)
    st.subheader("🔍 Hasil Prediksi:")
    st.write(f"**Penyakit:** {class_names[idx]}")
    st.write(f"**Akurasi Prediksi:** {prob:.2f}")

    # Probabilitas semua kelas
    st.write("### Detail Probabilitas:")
    for i, cls in enumerate(class_names):
        st.write(f"- {cls}: **{pred[0][i]:.4f}**")
