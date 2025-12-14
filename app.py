import streamlit as st
import joblib
import numpy as np
import pandas as pd # <-- Pastikan ini ada

# ... (Kode load_assets tetap sama) ...
# ... (Kode main() Streamlit tetap sama) ...

# ...

# Tombol Prediksi
if st.button("Lakukan Prediksi Diagnosis"):
    
    # ... (Validasi tetap sama) ...

    try:
        with st.spinner('Model sedang memproses...'):
            
            # --- PERBAIKAN KRITIS DI SINI ---
            # Model membutuhkan input DataFrame 2D dengan nama kolom yang sesuai.
            input_df = pd.DataFrame({'ciri_kasus': [input_text]})
            
            prediction_encoded = model_pipeline.predict(input_df)[0] # <-- Menggunakan input_df
            
            # Inverse Transform untuk mendapatkan nama diagnosis
            predicted_diagnosis = label_encoder.inverse_transform([prediction_encoded])[0]

        st.header("Hasil Prediksi")
        st.success(f"Diagnosis yang Diprediksi: **{predicted_diagnosis}**")
        
        # ... (Output tetap sama) ...

    except Exception as e:
        st.error(f"Gagal saat prediksi.")
        st.code(f"Error detail: {e}") 

# ...
