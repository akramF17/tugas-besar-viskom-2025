import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from PIL import Image

# Library tambahan untuk Real-time Webcam
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Postur Tangan", layout="wide")

# Load Model
@st.cache_resource
def load_model():
    model = YOLO('best.pt') 
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error memuat model: {e}")

# Sidebar & Judul
st.title("Proyek Visi Komputer: Deteksi Postur Tangan")
st.sidebar.header("Konfigurasi")

# Pilihan Input
source_option = st.sidebar.selectbox(
    "Pilih Sumber Data",
    ("Gambar (Upload)", "Video (Upload)", "Webcam (Real-Time Live)")
)

# Slider Confidence
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# LOGIKA 1: GAMBAR
if source_option == "Gambar (Upload)":
    uploaded_file = st.file_uploader("Upload Gambar", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        # Baca gambar
        image = Image.open(uploaded_file)
        
        with col1:
            st.image(image, caption='Gambar Asli', use_container_width=True)
        
        if st.button("Deteksi Objek"):
            # Prediksi
            res = model.predict(image, conf=conf_threshold)
            
            # Plot hasil (ini masih BGR)
            res_plotted = res[0].plot()
            
            # KONVERSI WARNA: BGR ke RGB
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.image(res_rgb, caption='Hasil Deteksi', use_container_width=True)

# LOGIKA 2: VIDEO
elif source_option == "Video (Upload)":
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Simpan sementara
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        vf = cv2.VideoCapture(tfile.name)
        
        # Placeholder untuk video
        stframe = st.empty()
        
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break
            
            # Prediksi
            results = model.predict(frame, conf=conf_threshold)
            res_plotted = results[0].plot()
            
            frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)
        
        vf.release()

# LOGIKA 3: WEBCAM REAL-TIME (WebRTC)
elif source_option == "Webcam (Real-Time Live)":
    st.write("Tunggu sebentar hingga webcam menyala. Pastikan browser mengizinkan akses kamera.")
    
    # Fungsi Callback untuk memproses setiap frame video secara langsung
    def video_frame_callback(frame):
        # Konversi frame WebRTC ke format OpenCV (numpy array)
        img = frame.to_ndarray(format="bgr24")
        
        # Lakukan Deteksi YOLO
        results = model.predict(img, conf=conf_threshold)
        
        # Gambar kotak anotasi pada frame
        annotated_frame = results[0].plot()
        
        # Kembalikan frame yang sudah dianotasi ke browser
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    # Jalankan Streamer
    webrtc_streamer(
        key="deteksi-isyarat", 
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
