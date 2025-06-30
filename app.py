import cv2
import math
import streamlit as st
from PIL import Image
import numpy as np

st.title("Deteksi Usia dan Gender")
st.write("Upload gambar untuk mendeteksi wajah, usia, dan gender.")


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

# --- Fungsi Highlight Wajah (Sama seperti kode asli Anda) ---
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

# --- Streamlit ---
uploaded_file = st.file_uploader("Pilih sebuah gambar", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Ketika file di-upload, kita perlu membacanya ke dalam format yang bisa digunakan OpenCV.
    # PIL (Pillow) digunakan untuk membuka file gambar, lalu kita konversi ke array NumPy.
    image = Image.open(uploaded_file)
    frame = np.array(image)
    # Konversi dari RGB (PIL) ke BGR (OpenCV)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Tampilkan gambar asli yang di-upload
    st.image(image, caption="Gambar yang Di-upload", use_column_width=True)
    st.write("Memproses...")

    # Proses gambar untuk deteksi
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
                
                # Prediksi Gender
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                
                # Prediksi Usia
                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                
                # Tampilkan hasil di gambar
                cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                
                # Tampilkan hasil dalam bentuk teks di bawah gambar
                st.success(f"Wajah Terdeteksi: Gender = {gender}, Perkiraan Usia = {age}")

            except Exception as e:
                st.error("Terjadi error saat memproses salah satu wajah. Mungkin wajah terlalu kecil atau di luar batas gambar.")
                print(e)
                continue
                
        st.image(cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB), caption='Gambar Hasil Deteksi', use_column_width=True)