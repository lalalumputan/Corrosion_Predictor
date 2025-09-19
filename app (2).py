import streamlit as st
import joblib
import numpy as np

# Load model & encoder
model = joblib.load("bagging_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("🔎 Prediksi Korosi Stainless Steel")
st.write("Masukkan parameter stainless steel dan kondisi lingkungan untuk memprediksi tingkat korosi.")

# --- Input Komposisi Kimia ---
st.header("⚙️ Komposisi Kimia (%)")
C = st.slider("Carbon (C)", 0.0, 2.0, 0.08)
Mn = st.slider("Manganese (Mn)", 0.0, 3.0, 1.0)
Cr = st.slider("Chromium (Cr)", 0.0, 30.0, 18.0)
Ni = st.slider("Nickel (Ni)", 0.0, 20.0, 8.0)
Mo = st.slider("Molybdenum (Mo)", 0.0, 5.0, 2.0)
Fe = st.slider("Iron (Fe)", 60.0, 90.0, 70.0)

# --- Input Lingkungan ---
st.header("🌍 Kondisi Lingkungan")
electrolytes = [
    "Sulfuric acid (H2SO4)",
    "Nitric acid (HNO3)",
    "Hydrochloric acid (HCl)",
    "Citric acid(HOC(CH2COOH)2COOH.H2O)",
    "KHSO4",
    "KNO3",
    "MgCl2.6H2O"
]
electrolyte = st.selectbox("Jenis Electrolyte", electrolytes)
temp = st.slider("Suhu (°C)", 20, 350, 25)

# --- Encode environment dummy ---
env_vector = [1 if e == electrolyte else 0 for e in electrolytes]

# --- Susun fitur input ---
X_input = np.array([[C, Mn, Cr, Ni, Mo, Fe] + env_vector + [temp]])

# --- Prediksi ---
if st.button("Prediksi"):
    pred = model.predict(X_input)[0]
    hasil = label_encoder.inverse_transform([pred])[0]  # bisa 4 kelas sesuai dataset
    st.success(f"📊 Prediksi Tingkat Korosi: **{hasil}**")

    # Probabilitas tiap kelas
    probs = model.predict_proba(X_input)[0]
    for cls, p in zip(label_encoder.classes_, probs):
        st.write(f"- {cls}: {p*100:.2f}%")

