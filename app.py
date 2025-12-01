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

st.title("🌾 Deteksi Penyakit Daun Padi Menggunakan CNN")
st.write("Upload gambar daun padi. Selain daun padi tidak akan diproses.")

uploaded = st.file_uploader("Upload gambar...", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    # ============================
    # 1. CEK WARNA (Validasi Daun)
    # ============================
    img_np = np.array(img)
    r_mean = np.mean(img_np[:,:,0])
    g_mean = np.mean(img_np[:,:,1])
    b_mean = np.mean(img_np[:,:,2])

    # Jika warna hijau tidak dominan → bukan daun padi
    if not (g_mean > r_mean and g_mean > b_mean):
        st.error("🚫 Gambar yang diupload **bukan daun padi**. Silakan upload daun padi.")
        st.image(img, caption="Gambar yang ditolak", use_column_width=True)
        st.stop()

    # ======================================
    # 2. Resize otomatis mengikuti model
    # ======================================
    input_shape = model.input_shape
    img_w = input_shape[1]
    img_h = input_shape[2]

    img_resized = img.resize((img_w, img_h))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    # ======================================
    # 3. Prediksi CNN
    # ======================================
    pred = model.predict(img_array)
    idx = np.argmax(pred)
    prob = np.max(pred)

    # ======================================
    # 4. Jika probabilitas rendah → bukan daun padi
    # ======================================
    if prob < 0.50:
        st.error("🚫 CNN mendeteksi bahwa ini **bukan daun padi**.")
        st.image(img, caption="Gambar yang ditolak", use_column_width=True)
        st.stop()

    # ======================================
    # 5. Output Hasil
    # ======================================
    st.image(img, caption="Gambar Input", use_column_width=True)
    st.subheader("🔍 Hasil Prediksi:")
    st.write(f"**Penyakit:** {class_names[idx]}")
    st.write(f"**Akurasi Prediksi:** {prob:.2f}")

    st.write("### Probabilitas Kelas:")
    for i, cls in enumerate(class_names):
        st.write(f"- {cls}: **{pred[0][i]:.4f}**")
