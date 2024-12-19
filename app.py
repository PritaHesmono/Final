import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Load machine learning model
model = joblib.load('model_balita_rbf.pkl')

# Fungsi untuk melakukan prediksi
def predict_stunting(nama_balita, tgl_lahir_balita, jenis_kelamin, berat_badan, panjang_badan):
    try:
        # Validasi data input
        if not (40 <= panjang_badan <= 200):
            return {'error': 'Tinggi badan tidak valid. Harus antara 40 dan 200 cm.'}
        if not (1 <= berat_badan <= 200):
            return {'error': 'Berat badan tidak valid. Harus antara 1 dan 200 kg.'}

        # Konversi jenis_kelamin ke numerik
        jenis_kelamin_perempuan = 0 if jenis_kelamin.lower() == 'perempuan' else 1

        # Hitung umur dalam bulan
        birth_date = datetime.strptime(tgl_lahir_balita, '%Y-%m-%d')
        current_date = datetime.now()
        age_in_months = (current_date.year - birth_date.year) * 12 + current_date.month - birth_date.month

        # Lakukan prediksi menggunakan model machine learning
        features = pd.DataFrame([{
            'Umur (bulan)': age_in_months,
            'Tinggi Badan (cm)': panjang_badan,
            'Jenis Kelamin_perempuan': jenis_kelamin_perempuan
        }])
        prediction = model.predict(features)

        # Buat response dengan hasil prediksi
        result = {
            'nama_balita': nama_balita,
            'hasil_prediksi': prediction[0]
        }

        return result

    except Exception as e:
        return {'error': str(e)}

# Streamlit UI
st.title("Prediksi Stunting Balita")

# Input form
nama_balita = st.text_input("Nama Balita")
tgl_lahir_balita = st.date_input("Tanggal Lahir Balita")
jenis_kelamin = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
berat_badan = st.number_input("Berat Badan (kg)", min_value=1.0, max_value=200.0)
panjang_badan = st.number_input("Panjang Badan (cm)", min_value=40.0, max_value=200.0)

if st.button("Prediksi"):
    # Melakukan prediksi
    result = predict_stunting(nama_balita, tgl_lahir_balita.strftime('%Y-%m-%d'), jenis_kelamin, berat_badan, panjang_badan)

    # Menampilkan hasil
    if 'error' in result:
        st.error(result['error'])
    else:
        st.success(f"Nama Balita: {result['nama_balita']}, Hasil Prediksi: {result['hasil_prediksi']}")