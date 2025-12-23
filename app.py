import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import av
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

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

# 3. Sidebar
st.title("Proyek Visi Komputer: Deteksi Postur Tangan")
st.sidebar.header("Konfigurasi")
source_option = st.sidebar.selectbox(
    "Pilih Sumber Data",
    ("Gambar (Upload)", "Video (Upload)", "Webcam (Real-Time Live)")
)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# ==========================================
# LOGIKA 1: GAMBAR
# ==========================================
if source_option == "Gambar (Upload)":
    uploaded_file = st.file_uploader("Upload Gambar", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption='Gambar Asli', use_container_width=True)
        
        if st.button("Deteksi Objek"):
            res = model.predict(image, conf=conf_threshold)
            res_plotted = res[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            with col2:
                st.image(res_rgb, caption='Hasil Deteksi', use_container_width=True)

# ==========================================
# LOGIKA 2: VIDEO
# ==========================================
elif source_option == "Video (Upload)":
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Simpan file upload ke temp
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        # Siapkan VideoWriter
        vf = cv2.VideoCapture(video_path)
        
        # Ambil properti video
        width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vf.get(cv2.CAP_PROP_FPS)
        
        # Path output sementara (.avi agar cepat ditulis OpenCV)
        output_path_temp = os.path.join(tempfile.gettempdir(), "temp_output.avi")
        # Path output final (.mp4 untuk browser)
        output_path_final = os.path.join(tempfile.gettempdir(), "final_output.mp4")

        # Codec MJPG untuk penulisan cepat sementara
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path_temp, fourcc, fps, (width, height))

        # Progress bar
        my_bar = st.progress(0)
        total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break
            
            # Deteksi
            results = model.predict(frame, conf=conf_threshold)
            res_plotted = results[0].plot()
            
            # Tulis ke video output
            out.write(res_plotted)
            
            frame_count += 1
            if total_frames > 0:
                my_bar.progress(min(frame_count / total_frames, 1.0))
        
        vf.release()
        out.release()
        my_bar.empty()

        # Konversi ke FFMPEG
        os.system(f"ffmpeg -y -i {output_path_temp} -vcodec libx264 {output_path_final}")

        # Tampilkan Video Hasil
        if os.path.exists(output_path_final):
            st.success("Selesai! Berikut hasilnya:")
            st.video(output_path_final)
        else:
            st.error("Gagal melakukan encoding video.")

# ==========================================
# LOGIKA 3: WEBCAM
# ==========================================
elif source_option == "Webcam (Real-Time Live)":
    st.write("Pastikan browser mengizinkan akses kamera.")
    
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Deteksi
        results = model.predict(img, conf=conf_threshold)
        annotated_frame = results[0].plot()
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    # Konfigurasi STUN Server (Penting untuk Cloud)
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="deteksi-isyarat",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False}
    )


