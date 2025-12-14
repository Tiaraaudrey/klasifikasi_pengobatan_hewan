import streamlit as st
import joblib
import numpy as np
MODEL_PATH = 'ai_diagnosa_pipeline.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

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

model_pipeline, label_encoder = load_assets()

# --- 3. Fungsi Utama Aplikasi Streamlit ---
def main():
    st.set_page_config(page_title="Diagnosa AI", layout="centered")

    st.title("Diagnosa AI: Klasifikasi Teks Kasus (Project 3)")
    st.markdown("""
        Masukkan ciri-ciri kasus (gejala) dalam bentuk teks untuk mendapatkan prediksi diagnosis dari model Klasifikasi NLP.
    """)
    st.markdown("---")

    # Input Teks dari User
    input_text = st.text_area(
        "**Masukkan Ciri-ciri Kasus (Gejala)**", 
        placeholder="Contoh: Demam, batuk, leleran hidung, dan ada pembengkakan pada kelenjar getah bening.",
        height=150
    )

    # Tombol Prediksi
    if st.button("Lakukan Prediksi Diagnosis"):
        
        if not input_text.strip():
            st.warning("Mohon masukkan teks ciri-ciri kasus.")
            return

        if model_pipeline is None or label_encoder is None:
            # Error ditangani di fungsi load_assets
            return

        try:
            with st.spinner('Model sedang memproses...'):
                # Core Prediction Logic
                # Menggunakan .predict pada pipeline (yang sudah termasuk TF-IDF dan Classifier)
                prediction_encoded = model_pipeline.predict([input_text])[0] 
                
                # Inverse Transform untuk mendapatkan nama diagnosis
                predicted_diagnosis = label_encoder.inverse_transform([prediction_encoded])[0]

            st.header("Hasil Prediksi")
            st.success(f"Diagnosis yang Diprediksi: **{predicted_diagnosis}**")
            
            st.markdown("---")
            st.info("Prediksi ini adalah output Machine Learning dan harus dikonfirmasi oleh profesional yang kompeten.")

        except Exception as e:
            st.error(f"Gagal saat prediksi. Pastikan format input benar. Error: {e}")

# Jalankan Aplikasi
if __name__ == "__main__":
    main()
