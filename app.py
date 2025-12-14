import streamlit as st
import joblib
import numpy as np

MODEL_PATH = 'model_assets/ai_diagnosa_pipeline.pkl'
LABEL_ENCODER_PATH = 'model_assets/label_encoder.pkl'

@st.cache_resource
def load_assets():
    """Memuat pipeline model dan label encoder dengan caching."""
    model_pipeline = None
    label_encoder = None
    
    try:
        model_pipeline = joblib.load(MODEL_FILE)
        label_encoder = joblib.load(LABEL_ENCODER_FILE)
        st.success("Aset model (Pipeline dan Encoder) berhasil dimuat.")
    except Exception as e:
        st.error(f"Error memuat aset model: {e}. Pastikan file {MODEL_FILE} dan {LABEL_ENCODER_FILE} ada di root directory GitHub.")
        
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
        
        # Validasi Input
        if not input_text.strip():
            st.warning("Mohon masukkan teks ciri-ciri kasus.")
            return

        if model_pipeline is None or label_encoder is None:
            st.error("Model gagal dimuat. Tidak dapat melakukan prediksi.")
            return

        try:
            with st.spinner('Model sedang memproses...'):
                # Core Prediction Logic
                # Model Pipeline menerima list of text
                prediction_encoded = model_pipeline.predict([input_text])[0] 
                
                # Inverse Transform untuk mendapatkan nama diagnosis
                predicted_diagnosis = label_encoder.inverse_transform([prediction_encoded])[0]

            st.header("Hasil Prediksi")
            st.success(f"Diagnosis yang Diprediksi: **{predicted_diagnosis}**")
            
            st.markdown("---")
            st.info("Prediksi ini dihasilkan oleh model machine learning dan harus dikonfirmasi oleh profesional yang kompeten.")

        except Exception as e:
            st.error(f"Gagal saat prediksi. Pastikan format input benar. Error: {e}")

# Jalankan Aplikasi
if __name__ == "__main__":
    main()

