import streamlit as st
import joblib
import numpy as np
import pandas as pd 
import re 

# --- 1. KONFIGURASI FILE MODEL & KOLOM ---
MODEL_PATH = 'ai_diagnosa_pipeline.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
ANIMAL_COL = 'Jenis_Hewan_Dominan' 
Y_COL = 'Diagnosa Banding'
TANGGAL_COL = 'tanggal_kasus' # Kolom tanggal untuk analisis TMI

# --- 2. Fungsi Memuat Aset (Model) ---
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
        st.error(f"FATAL ERROR: Gagal memuat aset model. Pastikan file {MODEL_PATH} dan {LABEL_ENCODER_PATH} ada.")
        st.code(f"Error detail: {e}")
        
    return model_pipeline, label_encoder

# --- 3. Fungsi Memuat dan Memproses Data Mentah (Untuk TMI) ---
def extract_jumlah_ekor(dosis_text):
    """Mengekstrak angka jumlah ekor dari string Dosis (default 1 jika tidak ditemukan)."""
    text = str(dosis_text).lower()
    # Mencari pola angka diikuti oleh 'ekor'
    match = re.search(r'(\d+)\s*ekor', text)
    if match:
        return int(match.group(1))
    else:
        # Jika tidak ada keterangan ekor, asumsikan 1 ekor/kasus
        return 1
        
def extract_animal(dosis_text):
    """Mengekstrak jenis hewan dari string Dosis."""
    text = str(dosis_text).lower()
    if re.search(r'sapi', text):
        return 'Sapi'
    elif re.search(r'kambing', text):
        return 'Kambing'
    elif re.search(r'kucing', text):
        return 'Kucing'
    elif re.search(r'anjing', text):
        return 'Anjing'
    else:
        return 'Lainnya'

# @st.cache_data
def load_raw_data():
    """Memuat dan membersihkan data untuk Analisis TMI."""
    try:
        # Muat semua file CSV data kasus
        df_obat_2022 = pd.read_csv('LAPORAN PENGOBATAN 2022.csv', sep=';')
        df_obat_2023 = pd.read_csv('LAPORAN PENGOBATAN 2023.csv', sep=';')
        df_obat_2024 = pd.read_csv('LAPORAN PENGOBATAN 2024.csv', sep=';')
        df_obat_2025 = pd.read_csv('LAPORAN PENGOBATAN 2025.csv', sep=';')
        df_kasus = pd.concat([df_obat_2022, df_obat_2023, df_obat_2024, df_obat_2025], ignore_index=True)
        
        # Pra-pemrosesan
        df_kasus['Dosis'] = df_kasus['Dosis'].fillna('')
        df_kasus[ANIMAL_COL] = df_kasus['Dosis'].apply(extract_animal)

        df_kasus['Jumlah Kasus'] = df_kasus['Dosis'].apply(extract_jumlah_ekor)
        
        # Pembersihan Target Y
        df_kasus.dropna(subset=[Y_COL], inplace=True)
        df_kasus = df_kasus[df_kasus[Y_COL].str.lower() != 'tidak sakit']
        df_kasus = df_kasus[df_kasus[Y_COL].str.lower() != '']
        
        # --- Konversi Kolom Tanggal dan Ekstraksi Tahun/Bulan ---
        if TANGGAL_COL in df_kasus.columns:
            df_kasus[TANGGAL_COL] = pd.to_datetime(df_kasus[TANGGAL_COL], errors='coerce')
            
            # Ekstrak TAHUN-BULAN untuk tren bulanan
            df_kasus['Tahun_Bulan'] = df_kasus[TANGGAL_COL].dt.strftime('%Y-%m')
            
            # Ekstrak TAHUN untuk metrik
            df_kasus['Tahun'] = df_kasus[TANGGAL_COL].dt.year.astype('Int64', errors='ignore').astype(str)
            
            df_kasus = df_kasus[df_kasus['Tahun'].notna() & (df_kasus['Tahun'] != '<NA>')].copy()
            df_kasus = df_kasus[df_kasus['Tahun_Bulan'].notna() & (df_kasus['Tahun_Bulan'] != 'NaT')].copy()
        else:
            st.warning(f"Kolom '{TANGGAL_COL}' tidak ditemukan. Tren Tahunan/Bulanan tidak akan muncul.")
            df_kasus['Tahun'] = 'N/A' 
            df_kasus['Tahun_Bulan'] = 'N/A' 
            
        # --- FILTERING KELAS MINORITAS ---
        min_class_count = 5 
        value_counts = df_kasus[Y_COL].value_counts()
        valid_classes = value_counts[value_counts >= min_class_count].index
        df_kasus = df_kasus[df_kasus[Y_COL].isin(valid_classes)].reset_index(drop=True).copy()

        return df_kasus
        
    except FileNotFoundError:
        st.warning("Salah satu atau semua File CSV data kasus tidak ditemukan. TMI tidak akan aktif.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Gagal memuat atau memproses data mentah. Detail error: {e}")
        return pd.DataFrame()

# --- 4a. Fungsi Menampilkan TMI HIGHLIGHTS (Di Samping) ---
def display_tmi_highlight(df):
    
    st.markdown("### 📊 Kilas Data Utama (TMI)")
    st.markdown("___")
    
    if df.empty:
        st.info("Data tidak tersedia.")
        return
        
    # Variabel Global dari TMI
    total_cases = df['Jumlah Kasus'].sum() 
    num_diagnoses = df[Y_COL].nunique()
    
    # Metrik
    st.metric("Total Ekor Diobati", f"{total_cases:,}")
    st.metric("Diagnosis Unik", f"{num_diagnoses}")

    # Top 5 Diagnosis (Ringkas dalam bentuk DataFrame)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Top 5 Diagnosis (Ekor)**")
    
    top_diagnosis = df.groupby(Y_COL)['Jumlah Kasus'].sum().sort_values(ascending=False).head(5)
    top_diagnosis_df = top_diagnosis.reset_index()
    top_diagnosis_df.columns = ['Diagnosis', 'Jumlah Kasus']
    st.dataframe(top_diagnosis_df, use_container_width=True, hide_index=True)


# --- 4b. Fungsi Menampilkan TMI TREN (Di Bawah) ---
def display_tmi_tren(df):
    
    st.markdown("## 📈 Tren Penyakit dari Bulan ke Bulan")
    st.markdown("---")
    
    if df.empty or 'Tahun_Bulan' not in df.columns or (df['Tahun_Bulan'] == 'N/A').all():
        st.info("Tren penyakit per bulan tidak dapat ditampilkan karena data tanggal tidak tersedia atau tidak valid.")
        return
        
    # 1. Identifikasi 5 diagnosis teratas secara keseluruhan (berdasarkan sum)
    top_5_diseases = df.groupby(Y_COL)['Jumlah Kasus'].sum().sort_values(ascending=False).head(5).index.tolist()
    
    # 2. Hitung jumlah kasus per tahun-bulan (Menggunakan SUM)
    df_trend = df[df[Y_COL].isin(top_5_diseases)].groupby(['Tahun_Bulan', Y_COL])['Jumlah Kasus'].sum().reset_index(name='Jumlah Kasus')
    
    # 3. Pivot data: Menggunakan 'Tahun_Bulan' sebagai index
    df_pivot = df_trend.pivot_table(index='Tahun_Bulan', columns=Y_COL, values='Jumlah Kasus', fill_value=0)
    
    # 4. Sorting data berdasarkan index (YYYY-MM)
    df_pivot.sort_index(inplace=True)
    
    # Tampilkan Line Chart
    st.line_chart(df_pivot) 
    st.markdown(f"""
    <p style='font-size: small; color: gray;'>
    Visualisasi di atas menunjukkan tren total ekor/kasus bulanan dari 5 diagnosis terbanyak ({', '.join(top_5_diseases)}).
    </p>
    """, unsafe_allow_html=True)
    
# --- 5. Fungsi Utama Aplikasi Streamlit (Main) ---
def main(model_pipeline, label_encoder, raw_df): 
    st.set_page_config(page_title="Prediksi Penyakit Hewan", layout="centered")

    st.title("Vet Diagnosa AI: Klasifikasi Penyakit Hewan")
    st.markdown("---")

    # --- MEMBUAT DUA KOLOM UTAMA (RASIO 3:2) ---
    col_prediksi, col_tmi_highlight = st.columns([3, 2]) 
    
    # -----------------------------------------------------
    # --- KOLOM KIRI: FORMULIR PREDIKSI ---
    with col_prediksi:
        st.header("Diagnosis Gejala (Model AI)")
        st.markdown("""
            **Tool Prediksi Diagnosis** menggunakan model *Machine Learning* yang dilatih dari data kasus dan gejala klinis.
        """)
        st.markdown("___")
        
        # --- Konstanta Hewan ---
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
            # ... (Semua logika prediksi tetap sama di sini) ...
            
            if not input_text.strip():
                st.warning("Mohon masukkan teks ciri-ciri kasus.")
                return

            if model_pipeline is None or label_encoder is None:
                return

            try:
                with st.spinner('Model sedang memproses...'):
                    
                    # 1. Konversi input ke DataFrame DUA KOLOM
                    input_df = pd.DataFrame({
                        'ciri_kasus': [input_text], 
                        ANIMAL_COL: [input_animal] 
                    })
                    
                    # 2. Prediksi
                    prediction_encoded = model_pipeline.predict(input_df)[0] 
                    
                    # 3. Inverse Transform
                    predicted_diagnosis = label_encoder.inverse_transform([prediction_encoded])[0]

                st.subheader("Hasil Prediksi")
                st.success(f"Diagnosis yang Diprediksi: **{predicted_diagnosis}**")
                st.info("Prediksi ini adalah output Machine Learning dan harus dikonfirmasi oleh profesional yang kompeten.")

            except Exception as e:
                st.error(f"Gagal saat prediksi.")
                st.code(f"Error detail: {e}") 
                
        # --- END KOLOM KIRI ---
    # -----------------------------------------------------
    
    # -----------------------------------------------------
    # --- KOLOM KANAN: HIGHLIGHT TMI ---
    with col_tmi_highlight:
        # Kita akan membuat fungsi baru atau memanggil display_tmi dengan mode 'highlight'
        display_tmi_highlight(raw_df) 
    # --- END KOLOM KANAN ---
    # -----------------------------------------------------

    st.markdown("___")
    # --- TAMPILAN TMI (TREN BESAR) DI BAWAH ---
    display_tmi_tren(raw_df)
# Jalankan Aplikasi
if __name__ == "__main__":
    # --- PEMUATAN ASET GLOBAL DAN EKSEKUSI MAIN ---
    model_pipeline, label_encoder = load_assets()
    raw_df = load_raw_data()
    main(model_pipeline, label_encoder, raw_df)




