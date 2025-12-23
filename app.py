import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import numpy as np
from PIL import Image

# Konfigurasi Halaman
st.set_page_config(
    page_title="Deteksi Postur Tangan",
    page_icon="🖐️",
    layout="wide"
)

# Load Model
@st.cache_resource
def load_model():
    model = YOLO('best.pt') 
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error memuat model: {e}")

# Judul & Deskripsi
st.title("Proyek Visi Komputer: Deteksi Postur Tangan")
st.markdown("""
Aplikasi ini menggunakan Computer Vision (YOLOv11) untuk mendeteksi postur tangan pada gambar dan video.
Silakan upload file Anda di menu sebelah kiri.
""")

st.divider()
st.subheader("Referensi Postur yang Dapat Dideteksi")
st.write("Berikut adalah 10 kelas postur tangan yang dilatih pada model ini:")

postur_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
cols = st.columns(5)

for i, postur in enumerate(postur_labels):
    col = cols[i % 5]
    
    with col:
        img_path = os.path.join("contoh_postur", f"{postur}.jpg")
        
        if os.path.exists(img_path):
            st.image(img_path, caption=f"Postur {postur.upper()}", use_container_width=True)
        else:
            st.warning(f"File {postur}.jpg tidak ditemukan di folder 'contoh_postur'")

st.divider()

# Sidebar Konfigurasi
st.sidebar.header("Panel Kontrol")
source_option = st.sidebar.selectbox(
    "Pilih Sumber Data",
    ("Gambar (Upload)", "Video (Upload)")
)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# ==========================================
# LOGIKA 1: GAMBAR (UPLOAD)
# ==========================================
if source_option == "Gambar (Upload)":
    st.subheader("Deteksi pada Gambar")
    uploaded_file = st.file_uploader("Upload Gambar", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file)
        
        with col1:
            st.image(image, caption='Gambar Asli', use_container_width=True)
        
        # Tombol Eksekusi
        if st.sidebar.button("Mulai Deteksi Gambar", type="primary"):
            with st.spinner('Sedang memproses...'):
                # Prediksi
                res = model.predict(image, conf=conf_threshold)
                
                # Plotting
                res_plotted = res[0].plot()
                
                # Konversi BGR (OpenCV) ke RGB (Streamlit)
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.image(res_rgb, caption='Hasil Deteksi', use_container_width=True)
                    st.success("Selesai!")

# ==========================================
# LOGIKA 2: VIDEO (UPLOAD)
# ==========================================
elif source_option == "Video (Upload)":
    st.subheader("Deteksi pada Video")
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Simpan file sementara
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        # Tombol Eksekusi
        if st.sidebar.button("Mulai Deteksi Video", type="primary"):
            
            # Setup Video Processing
            vf = cv2.VideoCapture(video_path)
            width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = vf.get(cv2.CAP_PROP_FPS)
            total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Output Paths
            output_path_temp = os.path.join(tempfile.gettempdir(), "temp_output.avi")
            output_path_final = os.path.join(tempfile.gettempdir(), "final_output.mp4")

            # Writer (MJPG for speed)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(output_path_temp, fourcc, fps, (width, height))

            # UI Progress
            st.write("Sedang memproses video frame demi frame...")
            progress_bar = st.progress(0)
            frame_count = 0

            # Loop Processing
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret:
                    break
                
                # Prediksi
                results = model.predict(frame, conf=conf_threshold)
                res_plotted = results[0].plot()
                
                # Write to temp video
                out.write(res_plotted)
                
                # Update Progress
                frame_count += 1
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
            
            # Cleanup resources
            vf.release()
            out.release()
            progress_bar.empty()

            # Konversi FFMPEG (Agar bisa diputar di browser)
            st.info("Melakukan encoding video akhir (H.264)...")
            os.system(f"ffmpeg -y -i {output_path_temp} -vcodec libx264 {output_path_final}")

            # Tampilkan Hasil
            if os.path.exists(output_path_final):
                st.success("Proses Selesai! Berikut hasilnya:")
                st.video(output_path_final)
            else:
                st.error("Gagal melakukan encoding video. Pastikan ffmpeg terinstal.")
