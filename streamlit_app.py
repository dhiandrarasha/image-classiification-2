import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import zipfile
import io
import gc
import urllib.request

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Deteksi Keretakan Beton - CNN ZIP",
    page_icon="🔍",
    layout="centered"
)

# Informasi Pembuat di Sidebar (Sesuai file asli Anda)
st.sidebar.markdown("### Profil Pengembang")
st.sidebar.write("**Nama:** Dhiandra Rasha Zaputra")
st.sidebar.write("**NIM:** 032400040")
st.sidebar.write("**Jurusan:** ELEKTRO MEKANIKA 2024")

# Nama berkas model default sesuai skrip Anda
MODEL_FILE = "model_crack_beton.h5"

# =========================================================================
# PASTE LINK GOOGLE DRIVE KAMU DI BAWAH INI (Ganti teks di dalam tanda kutip)
# =========================================================================
GDrive_Link = "https://drive.google.com/file/d/1V_JHGbkansjGbsbCJKaZHY9t7Ux7IW51/view?usp=sharing"

# Fungsi untuk mengubah link share Google Drive biasa menjadi link download langsung
def get_direct_download_link(url):
    if "drive.google.com" in url:
        if "/file/d/" in url:
            file_id = url.split("/file/d/")[1].split("/")[0]
            return f"https://docs.google.com/uc?export=download&id={file_id}"
    return url

# 2. Memuat Model Terlatih via Google Drive (Disimpan di Cache)
@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_FILE):
        if GDrive_Link == "MASUKKAN_LINK_GOOGLE_DRIVE_KAMU_DI_SINI" or GDrive_Link == "":
            st.warning("⚠️ Kamu belum memasukkan link Google Drive di dalam kode script!")
            return None
        
        with st.spinner("⏳ Mengunduh file model h5 dari Google Drive (Hanya dilakukan sekali saat pertama kali dibuka)..."):
            try:
                direct_link = get_direct_download_link(GDrive_Link)
                urllib.request.urlretrieve(direct_link, MODEL_FILE)
                st.success("✅ Model berhasil diunduh dari Google Drive!")
            except Exception as e:
                st.error(f"❌ Gagal mengunduh model dari Google Drive: {e}")
                return None
                
    try:
        return tf.keras.models.load_model(MODEL_FILE)
    except Exception as e:
        st.error(f"Gagal memuat file model h5: {e}")
        return None

model = load_model_from_drive()

# Daftar kelas sesuai struktur dataset Anda
class_names = ['Retak', 'Tidak_Retak']

# 3. Antarmuka Pengguna Utama
st.title("🔍 Aplikasi Klasifikasi Gambar: Deteksi Keretakan Beton via ZIP")
st.write("Unggah berkas **.zip** berisi kumpulan foto permukaan beton untuk menganalisis struktur retak secara massal.")

if model is not None:
    st.success("✅ Model AI berhasil dimuat dan siap digunakan!")

# Komponen Unggah Berkas ZIP
uploaded_zip = st.file_uploader("Pilih berkas ZIP berisi kumpulan foto beton...", type=["zip"])

if uploaded_zip is not None and model is not None:
    st.write("---")
    st.info("📦 Berkas ZIP terdeteksi! Mengekstrak isi file...")
    
    try:
        with zipfile.ZipFile(uploaded_zip) as z:
            all_files = z.namelist()
            valid_extensions = ('.jpg', '.jpeg', '.png')
            
            # Memfilter file gambar asli dan membuang file sampah sistem (seperti metadata Mac/Windows)
            image_files = [
                f for f in all_files 
                if f.lower().endswith(valid_extensions) 
                and not f.startswith('__MACOSX/') 
                and not os.path.basename(f).startswith('.')
            ]
            
            total_images = len(image_files)
            
            if total_images == 0:
                st.warning("⚠️ Tidak ditemukan file gambar (.jpg/.png) yang valid di dalam ZIP Anda.")
            else:
                st.success(f"🚀 Menemukan {total_images} gambar. Memulai klasifikasi otomatis...")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []

                # Proses gambar satu per satu dari file ZIP
                for idx, file_name in enumerate(image_files):
                    status_text.text(f"Menganalisis ({idx + 1}/{total_images}): {os.path.basename(file_name)}")
                    
                    try:
                        img_data = z.read(file_name)
                        with Image.open(io.BytesIO(img_data)) as image:
                            # 4. Pra-proses Gambar (Sesuai spesifikasi training Anda: 150x150)
                            img_height, img_width = 150, 150
                            image_resized = image.resize((img_width, img_height))
                            
                            # Konversi ke NumPy array
                            img_array = np.array(image_resized)
                            if img_array.shape[-1] == 4:  # Bersihkan channel alpha jika PNG transparan
                                img_array = img_array[:, :, :3]
                            
                            # Tambahkan dimensi batch
                            img_array = np.expand_dims(img_array, axis=0)
                        
                        # 5. Melakukan Prediksi Inferensi
                        predictions = model.predict(img_array, verbose=0)
                        
                        # Logika penentuan kelas (Meniru persis kode prediksi di file asli Anda)
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
                        
                        # Teks status tampilan tabel
                        status_tabel = "🚨 RETAK" if hasil_prediksi == 'Retak' else "✅ AMAN"
                        
                        results.append({
                            "Nama File": os.path.basename(file_name),
                            "Status Analisis": status_tabel,
                            "Tingkat Kepercayaan": f"{konfidensi:.2f}%"
                        })
                        
                    except Exception:
                        continue
                    
                    # Perbarui Progress Bar
                    progress_bar.progress((idx + 1) / total_images)
                    
                    # Bersihkan RAM secara berkala agar Streamlit Cloud tidak kehabisan memori
                    if (idx + 1) % 5 == 0:
                        tf.keras.backend.clear_session()
                        gc.collect()

                status_text.empty()
                
                # 6. Tampilkan Hasil Akhir Massal
                if results:
                    st.write("### 📊 Hasil Klasifikasi Keseluruhan:")
                    st.dataframe(results, use_container_width=True)
                    
                    total_pos = sum(1 for r in results if "RETAK" in r["Status Analisis"])
                    total_neg = sum(1 for r in results if "AMAN" in r["Status Analisis"])
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Total Terdeteksi Retak (Cracked)", total_pos)
                    col2.metric("Total Terdeteksi Aman (Normal)", total_neg)
                        
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat membuka berkas ZIP: {e}")

st.markdown("---")
st.caption("Aplikasi berbasis Web ini dikembangkan menggunakan kerangka kerja Streamlit dan backend TensorFlow (Keras).")
