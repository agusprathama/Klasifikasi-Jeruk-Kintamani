import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from skimage.feature import graycomatrix, graycoprops
import os
import matplotlib.pyplot as plt

def extract_features_from_image(image):
    crop = image[90:510, 199:618]
    resize = cv2.resize(crop, (800,600))  
    hsv = cv2.cvtColor(resize, cv2.COLOR_BGR2HSV)
    avg_color = np.mean(hsv, axis=(0, 1))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)
        x, y, w, h = cv2.boundingRect(max_contour)
        aspect_ratio = float(w) / h
        perimeter = cv2.arcLength(max_contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
    else:
        area, aspect_ratio, circularity = 0, 0, 0

    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]

    return [avg_color[0], avg_color[1], avg_color[2], area, aspect_ratio, circularity, contrast, homogeneity, energy], {
            'Hue': f"{round(avg_color[0])}°",
            'Saturation': f"{round(avg_color[1])}%",
            'Value': f"{round(avg_color[2])}%",
            'Luas Area': f"{int(area)} px²",
            'Aspect Ratio': f"{round(aspect_ratio, 2)} (rasio)",
            'Circularity': f"{round(circularity, 2)} (rasio)",
            'Kontras': f"{round(contrast, 2)}",
            'Homogenitas': f"{round(homogeneity, 2)}",
            'Energi': f"{round(energy, 2)}"
    }


base_path = os.path.dirname(__file__)
knn = joblib.load(os.path.join(base_path, 'model_knn.pkl'))
label_encoder = joblib.load(os.path.join(base_path, 'label_encoder.pkl'))

st.title("🍊 Aplikasi Klasifikasi Mutu Jeruk Kintamani")
st.write("Upload gambar jeruk untuk memprediksi mutu (A, B, atau C).")

uploaded_files = st.file_uploader("Pilih satu atau lebih gambar", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    hasil = []
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        features, feature_dict = extract_features_from_image(image)

        pred = knn.predict(np.array(features).reshape(1, -1))
        mutu = label_encoder.inverse_transform(pred)[0]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, channels="BGR", caption=f"Gambar: {uploaded_file.name}", use_container_width=True)

        with col2:
            st.markdown(f"### ✅ Prediksi Mutu: **{mutu}**")
            st.markdown("**Fitur Ekstraksi:**")
            df_fitur = pd.DataFrame(feature_dict.items(), columns=["Fitur", "Nilai"])
            st.table(df_fitur)


  

        hasil.append({"Nama File": uploaded_file.name, "Prediksi Mutu": mutu})

    df_hasil = pd.DataFrame(hasil)
    st.markdown("### 📄 Hasil Keseluruhan")
    st.dataframe(df_hasil)

    csv = df_hasil.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Hasil ke CSV", data=csv, file_name="hasil_prediksi_jeruk.csv", mime="text/csv")
