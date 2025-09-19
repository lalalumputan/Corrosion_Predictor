import streamlit as st
import joblib
import numpy as np

# Load model & encoder
model = joblib.load("bagging_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("🔎 Prediksi Korosi Stainless Steel")
st.write("Masukkan parameter untuk memprediksi tingkat korosi.")

# --- Input fitur sederhana (contoh minimal dulu) ---
C = st.slider("Kadar Carbon (%)", 0.0, 2.0, 0.08)
Mn = st.slider("Kadar Mangan (%)", 0.0, 3.0, 1.0)
Cr = st.slider("Kadar Chromium (%)", 0.0, 30.0, 18.0)

electrolyte = st.selectbox("Jenis Electrolyte", label_encoder.classes_)
encoded_env = label_encoder.transform([electrolyte])[0]

X_input = np.array([[C, Mn, Cr, encoded_env]])

if st.button("Prediksi"):
    pred = model.predict(X_input)[0]
    hasil = label_encoder.inverse_transform([pred])[0]
    st.success(f"Hasil Prediksi: {hasil}")

