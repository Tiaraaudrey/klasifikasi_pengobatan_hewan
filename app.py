import streamlit as st
import joblib
import numpy as np
import pandas as pd # Wajib untuk membuat DataFrame input

# --- 1. KONFIGURASI FILE MODEL (Jalur Diperbaiki ke Root) ---
MODEL_PATH = 'ai_diagnosa_pipeline.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

# --- 2. Fungsi Memuat Model ---
# Kita gunakan st.cache_resource untuk memastikan model hanya dimuat sekali
@st.cache_resource
def load_assets():
    """Memuat pipeline model dan label encoder dari root directory."""
    model_pipeline = None
    label_encoder = None
    
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        st.success("Aset model (Pipeline dan Encoder) berhasil dimuat.")
    except Exception as e:
        st.error(f"FATAL ERROR: Gagal memuat aset model. Pastikan file {MODEL_PATH} dan {LABEL_ENCODER_PATH} ada di FOLDER UTAMA repositori GitHub Anda.")
        st.code(f"Error detail: {e}")
        
    return model_pipeline, label_encoder

# --- 3. Fungsi Utama Aplikasi Streamlit ---
def main(model_pipeline, label_encoder): # Menerima model sebagai argumen
    st.set_page_config(page_title="Prediksi Penyakit Hewan", layout="centered")

    st.title("Diagnosa Penyakit Hewan melalui Gejala")
    st.markdown("""
        Masukkan ciri-ciri kasus (gejala) dan jenis hewan untuk mendapatkan prediksi diagnosis dari model Klasifikasi NLP.
    """)
    st.markdown("---")
    
    # --- Konstanta Hewan ---
    ANIMAL_COL = 'Jenis_Hewan_Dominan' # Harus sama persis dengan nama kolom di notebook Cell 5!
    
    # Daftar Jenis Hewan yang diekstrak (Harus sesuai dengan yang ditemukan di Cell 1 notebook)
    animal_list = ['Sapi', 'Kambing', 'Kucing', 'Anjing', 'Lainnya'] 

    # --- Input Teks dari User ---
    input_text = st.text_area(
        "**1. Masukkan Ciri-ciri Kasus (Gejala)**", 
        placeholder="Contoh: Demam, batuk, leleran hidung, dan ada pembengkakan pada kelenjar getah bening.",
        height=150
    )

    # --- Input Jenis Hewan ---
    input_animal = st.selectbox(
        f"**2. Pilih Jenis Hewan**",
        options=animal_list
    )

    # Tombol Prediksi
    if st.button("Lakukan Prediksi Diagnosis"):
        
        if not input_text.strip():
            st.warning("Mohon masukkan teks ciri-ciri kasus.")
            return

        if model_pipeline is None or label_encoder is None:
            # Notifikasi error sudah ditampilkan di load_assets
            return

        try:
            with st.spinner('Model sedang memproses...'):
                
                # 1. Konversi input ke DataFrame DUA KOLOM
                input_df = pd.DataFrame({
                    'ciri_kasus': [input_text], 
                    ANIMAL_COL: [input_animal] # Menambahkan kolom Jenis Hewan
                })
                
                # 2. Prediksi
                prediction_encoded = model_pipeline.predict(input_df)[0] 
                
                
                # 3. Inverse Transform
                predicted_diagnosis = label_encoder.inverse_transform([prediction_encoded])[0]

            st.header("Hasil Prediksi")
            st.success(f"Diagnosis yang Diprediksi: **{predicted_diagnosis}**")
            
            st.markdown("---")
            st.info("Prediksi ini adalah output Machine Learning dan harus dikonfirmasi oleh profesional yang kompeten.")

        except Exception as e:
            st.error(f"Gagal saat prediksi.")
            st.code(f"Error detail: {e}") 

# Jalankan Aplikasi
if __name__ == "__main__":
    # --- 4. PEMUATAN ASET GLOBAL DAN EKSEKUSI MAIN ---
    model_pipeline, label_encoder = load_assets()
    main(model_pipeline, label_encoder)
