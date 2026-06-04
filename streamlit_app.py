import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Deteksi Keretakan Beton - CNN",
    page_icon="🔍",
    layout="centered"
)

# Informasi Pembuat di Sidebar (Berdasarkan Metadata Notebook Anda)
st.sidebar.markdown("### Profil Pengembang")
st.sidebar.write("**Nama:** Dhiandra Rasha Zaputra")
st.sidebar.write("**NIM:** 032400040")
st.sidebar.write("**Jurusan:** ELEKTRO MEKANIKA 2024")

# 2. Memuat Model Terlatih (Disimpan di Cache agar tidak reload terus-menerus)
@st.cache_resource
def load_my_model():
    import os
    possible_names = ['model_crack_beton.h5', 'model_klasifikasi_retak.h5']
    for name in possible_names:
        if os.path.exists(name):
            return tf.keras.models.load_model(name), name
            
    try:
        return tf.keras.models.load_model('model_crack_beton.h5'), 'model_crack_beton.h5'
    except:
        return None, 'model_crack_beton.h5'

model, model_filename = load_my_model()

# Daftar kelas sesuai struktur dataset Anda
class_names = ['Retak', 'Tidak_Retak']

# 3. Antarmuka Pengguna Utama
st.title("🔍 Aplikasi Klasifikasi Gambar: Deteksi Keretakan Beton")
st.write("Unggah gambar permukaan beton untuk menganalisis dan mendeteksi adanya struktur retak.")

# Peringatan jika file model h5 belum diletakkan di satu folder aplikasi
if model is None:
    st.warning(f"⚠️ File model `{model_filename}` tidak ditemukan di direktori aplikasi. Harap letakkan file `.h5` hasil training Anda di folder yang sama dengan file `streamlit_app.py` agar aplikasi berfungsi.")

# Komponen Unggah Gambar
uploaded_file = st.file_uploader("Pilih file gambar beton...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membuka gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar Beton yang Diunggah', use_column_width=True)
    
    # Tombol Aksi untuk Memulai Prediksi
    if st.button("Mulai Deteksi 🚀"):
        if model is None:
            st.error("Gagal mendeteksi: Model belum dimuat secara benar.")
        else:
            with st.spinner('Sedang memproses gambar dan menjalankan model CNN...'):
                try:
                    # 4. Pra-proses Gambar (Menyesuaikan dengan spesifikasi training: 150x150)
                    img_height, img_width = 150, 150
                    image_resized = image.resize((img_width, img_height))
                    
                    # Konversi ke NumPy array dan pastikan format 3 channel warna (RGB)
                    img_array = np.array(image_resized)
                    if img_array.shape[-1] == 4:  # Bersihkan channel alpha jika format PNG transparan
                        img_array = img_array[:, :, :3]
                    
                    # Tambahkan dimensi batch (dari [150, 150, 3] menjadi [1, 150, 150, 3])
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # 5. Melakukan Prediksi Inferensi
                    predictions = model.predict(img_array)
                    
                    # Logika penentuan kelas (Meniru persis kode prediksi di notebook Anda)
                    if len(class_names) == 2 and predictions.shape[-1] == 1:
                        # Kasus Output 1 Neuron (Sigmoid Aktivasi)
                        predicted_class_idx = 1 if predictions[0][0] >= 0.5 else 0
                        konfidensi = predictions[0][0] if predicted_class_idx == 1 else 1 - predictions[0][0]
                        konfidensi = konfidensi * 100
                    else:
                        # Kasus Output Multi-neuron / Softmax Aktivasi
                        score = tf.nn.softmax(predictions[0])
                        predicted_class_idx = np.argmax(predictions[0])
                        konfidensi = np.max(score) * 100
                        
                    hasil_prediksi = class_names[predicted_class_idx]
                    
                    # 6. Tampilkan Visualisasi Hasil Akhir
                    st.subheader("Hasil Analisis:")
                    if hasil_prediksi == 'Retak':
                        st.error(f"🚨 **STATUS: TERDETEKSI RETAK (CRACKED)**")
                    else:
                        st.success(f"✅ **STATUS: AMAN / TIDAK RETAK (NORMAL)**")
                        
                    st.metric(label="Tingkat Kepercayaan (Confidence)", value=f"{konfidensi:.2f}%")
                    
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

st.markdown("---")
st.caption("Aplikasi berbasis Web ini dikembangkan menggunakan kerangka kerja Streamlit dan backend TensorFlow (Keras).")
