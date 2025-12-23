import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from PIL import Image
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Bahasa Isyarat BISINDO", layout="wide")

# Load Model 
@st.cache_resource
def load_model():
    model = YOLO('best.pt') 
    return model

try:
    model = load_model()
    st.sidebar.success("Model berhasil dimuat!")
except Exception as e:
    st.sidebar.error(f"Error memuat model: {e}")

# Judul & Sidebar
st.title("Proyek Visi Komputer: Deteksi Bahasa Isyarat")
st.sidebar.header("Pilih Metode Input")
source_option = st.sidebar.selectbox(
    "Sumber Data",
    ("Gambar (Upload)", "Video (Upload)", "Webcam (Live Snapshot)")
)

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# Logika Deteksi
if source_option == "Gambar (Upload)":
    uploaded_file = st.file_uploader("Upload Gambar", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar Asli', use_column_width=True)
        
        if st.button("Deteksi Objek"):
            res = model.predict(image, conf=conf_threshold)
            res_plotted = res[0].plot()
            st.image(res_plotted, caption='Hasil Deteksi', use_column_width=True)

elif source_option == "Video (Upload)":
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    if uploaded_file is not None:
        # Simpan file sementara agar bisa dibaca OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        vf = cv2.VideoCapture(tfile.name)
        stframe = st.empty() # Placeholder untuk menampilkan frame video
        
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break
            
            # Deteksi
            results = model.predict(frame, conf=conf_threshold)
            res_plotted = results[0].plot()
            
            # Konversi warna BGR ke RGB untuk Streamlit
            frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)
        
        vf.release()

elif source_option == "Webcam (Live Snapshot)":
    st.write("Ambil foto langsung menggunakan webcam untuk dideteksi.")
    img_file_buffer = st.camera_input("Ambil Gambar")
    
    if img_file_buffer is not None:
        # Konversi ke format yang bisa dibaca OpenCV/YOLO
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        results = model.predict(cv2_img, conf=conf_threshold)
        res_plotted = results[0].plot()
        
        # Tampilkan hasil (Convert BGR to RGB)
        st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption="Hasil Deteksi Webcam")