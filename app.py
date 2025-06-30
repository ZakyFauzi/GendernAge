import cv2
import streamlit as st
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

st.title("Deteksi Usia dan Gender")
st.write("Aplikasi ini dapat mendeteksi usia dan gender dari gambar yang di-upload atau melalui kamera real-time.")

@st.cache_resource
def load_models():
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    return faceNet, ageNet, genderNet

faceNet, ageNet, genderNet = load_models()

# Definisikan list dan nilai konstan
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Laki-laki', 'Perempuan']
padding = 20

# --- Fungsi Highlight Wajah (Tidak berubah) ---
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

app_mode = st.sidebar.selectbox('Pilih Mode Aplikasi',
    ['Tentang Aplikasi', 'Upload Gambar', 'Kamera Real-time']
)

if app_mode == 'Tentang Aplikasi':
    st.markdown("Aplikasi ini dibuat untuk mendemonstrasikan deteksi Wajah, Gender, dan Usia menggunakan OpenCV dan Streamlit.")
    st.markdown("""
    - **Upload Gambar:** Mode ini memungkinkan Anda mengunggah gambar dari perangkat Anda untuk dianalisis.
    - **Kamera Real-time:** Mode ini akan mengakses kamera Anda untuk melakukan deteksi secara langsung. Izinkan akses kamera jika diminta oleh browser.
    """)

elif app_mode == 'Upload Gambar':
    # --- Logika untuk Upload File (Sama seperti sebelumnya) ---
    uploaded_file = st.file_uploader("Pilih sebuah gambar", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        st.image(image, caption="Gambar yang Di-upload", use_column_width=True)
        st.write("Memproses...")

        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:
            st.warning("Tidak ada wajah yang terdeteksi.")
        else:
            for faceBox in faceBoxes:
                face = frame[max(0, faceBox[1] - padding):
                           min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                           :min(faceBox[2] + padding, frame.shape[1] - 1)]
                try:
                    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                    genderNet.setInput(blob)
                    genderPreds = genderNet.forward()
                    gender = genderList[genderPreds[0].argmax()]
                    
                    ageNet.setInput(blob)
                    agePreds = ageNet.forward()
                    age = ageList[agePreds[0].argmax()]
                    
                    cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    st.success(f"Wajah Terdeteksi: Gender = {gender}, Perkiraan Usia = {age}")
                except Exception:
                    continue
            st.image(cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB), caption='Gambar Hasil Deteksi', use_column_width=True)

elif app_mode == 'Kamera Real-time':
    st.header("Deteksi Real-time via Kamera")
    st.write("Klik 'START' untuk memulai deteksi. Izinkan browser untuk mengakses kamera Anda.")

    # --- Class untuk memproses video dari WebRTC ---
    class AgeGenderTransformer(VideoTransformerBase):
        def transform(self, frame):
            # Konversi frame dari format webrtc ke format opencv
            img = frame.to_ndarray(format="bgr24")
            
            # Lakukan proses deteksi yang sama persis dengan mode upload
            resultImg, faceBoxes = highlightFace(faceNet, img)
            if faceBoxes:
                for faceBox in faceBoxes:
                    face = img[max(0, faceBox[1] - padding):
                               min(faceBox[3] + padding, img.shape[0] - 1), max(0, faceBox[0] - padding)
                               :min(faceBox[2] + padding, img.shape[1] - 1)]
                    try:
                        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                        genderNet.setInput(blob)
                        genderPreds = genderNet.forward()
                        gender = genderList[genderPreds[0].argmax()]
                        
                        ageNet.setInput(blob)
                        agePreds = ageNet.forward()
                        age = ageList[agePreds[0].argmax()]
                        
                        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    except Exception:
                        continue
            
            return resultImg

    webrtc_streamer(key="example", 
                    video_transformer_factory=AgeGenderTransformer,
                    rtc_configuration=RTC_CONFIGURATION,
                    media_stream_constraints={"video": True, "audio": False})
