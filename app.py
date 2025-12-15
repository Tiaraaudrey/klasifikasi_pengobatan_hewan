import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re # Diperlukan untuk ekstraksi hewan

# --- 1. KONFIGURASI FILE MODEL (Jalur Diperbaiki ke Root) ---
MODEL_PATH = 'ai_diagnosa_pipeline.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
ANIMAL_COL = 'Jenis_Hewan_Dominan'
Y_COL = 'Diagnosa Banding'

# --- 2. Fungsi Memuat Model ---
def load_assets():
    """Memuat pipeline model dan label encoder dari root directory."""
    # st.cache_resource digunakan untuk mencegah pemuatan ulang yang tidak perlu
    # Namun, saat debugging, sebaiknya jangan gunakan cache.
    # Setelah yakin model sudah benar, Anda bisa menambahkan @st.cache_resource lagi.
    
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

# --- 3. Fungsi Memuat dan Memproses Data Mentah (Untuk TMI) ---
def extract_animal(dosis_text):
    """Mengekstrak jenis hewan dari string Dosis (sama seperti di Cell 1 notebook)."""
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
        # Ganti dengan nama file CSV Anda
        df_obat_2022 = pd.read_csv('LAPORAN PENGOBATAN 2022.csv', sep=';')
        df_obat_2023 = pd.read_csv('LAPORAN PENGOBATAN 2023.csv', sep=';')
        df_obat_2024 = pd.read_csv('LAPORAN PENGOBATAN 2024.csv', sep=';')
        df_obat_2025 = pd.read_csv('LAPORAN PENGOBATAN 2025.csv', sep=';')
        df_kasus = pd.concat([df_obat_2022, df_obat_2023, df_obat_2024, df_obat_2025], ignore_index=True)
        
        # Pra-pemrosesan (Wajib sama seperti di notebook Cell 1)
        df_kasus['Tanda/Sindrom'] = df_kasus['Tanda/Sindrom'].fillna('')
        df_kasus['Dosis'] = df_kasus['Dosis'].fillna('')
        df_kasus[ANIMAL_COL] = df_kasus['Dosis'].apply(extract_animal)
        
        # Pembersihan Target Y
        df_kasus.dropna(subset=[Y_COL], inplace=True)
        df_kasus = df_kasus[df_kasus[Y_COL].str.lower() != 'tidak sakit']
        df_kasus = df_kasus[df_kasus[Y_COL].str.lower() != '']

        # --- FIX KRUSIAL: Konversi Kolom Tanggal dan Ekstraksi Tahun ---
        TANGGAL_COL = 'tanggal_kasus' # Gunakan kolom ini sesuai output notebook
        
        if TANGGAL_COL in df_kasus.columns:
            # Mengubah kolom tanggal menjadi datetime, error='coerce' akan mengubah yang gagal menjadi NaT
            df_kasus[TANGGAL_COL] = pd.to_datetime(df_kasus[TANGGAL_COL], errors='coerce')
            
            # Ekstrak Tahun dari tanggal
            df_kasus['Tahun'] = df_kasus[TANGGAL_COL].dt.year.astype('Int64', errors='ignore').astype(str)
            
            # Hapus baris yang tahunnya tidak valid setelah konversi
            df_kasus = df_kasus[df_kasus['Tahun'] != '<NA>'].copy()
        else:
            # Jika kolom tanggal_kasus tidak ada
            st.warning(f"Kolom '{TANGGAL_COL}' tidak ditemukan di data mentah. Tren Tahunan tidak akan muncul.")
            df_kasus['Tahun'] = 'N/A' # Tambahkan kolom dummy
            
        # --- FILTERING KELAS MINORITAS (Sama seperti di notebook) ---
        min_class_count = 5 
        value_counts = df_kasus[Y_COL].value_counts()
        valid_classes = value_counts[value_counts >= min_class_count].index
        df_kasus = df_kasus[df_kasus[Y_COL].isin(valid_classes)].reset_index(drop=True).copy()
        
        return df_kasus
        
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Gagal memuat atau memproses data mentah: {e}")
        return pd.DataFrame()

# --- 4. Fungsi Menampilkan TMI (Insights) ---
def display_tmi(df):
    
    st.markdown("## 📈 TMI: Tren Penyakit dan Data Penting")
    st.markdown("---")
    
    if df.empty:
        st.info("File data CSV tidak ditemukan di folder utama (root). Analisis tidak dapat ditampilkan.")
        return
        
    # Asumsi kolom 'tanggal_kasus' ada di data Anda
    try:
        df['Tahun'] = df['tanggal_kasus'].astype(str).str[:4]
        
        # --- Row 1: Metrik Utama ---
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Total Sampel Data", f"{len(df):,}")
        
        num_diagnoses = df[Y_COL].nunique()
        col2.metric("Jumlah Total Diagnosis Unik", f"{num_diagnoses}")

        num_years = df['Tahun'].nunique()
        col3.metric("Jangkauan Tahun Data", f"{num_years} Tahun")

    except KeyError:
        st.error("Kolom 'tanggal_kasus' tidak ditemukan. Analisis Tren Tahunan tidak dapat ditampilkan.")
        return # Hentikan fungsi jika kolom krusial hilang

    st.markdown("---")

    # --- Row 2: Top 5 Diagnosis Terbanyak (Tetap Dipertahankan) ---
    col4, col5 = st.columns([1, 2]) # Sesuaikan lebar kolom
    
    # 1. Diagnosis Paling Sering
    top_diagnosis = df[Y_COL].value_counts().head(5)
    with col4:
        st.subheader("Top 5 Diagnosis")
        top_diagnosis_df = top_diagnosis.reset_index()
        top_diagnosis_df.columns = ['Diagnosis', 'Jumlah Kasus']
        st.dataframe(top_diagnosis_df, use_container_width=True, hide_index=True)

    # 2. Distribusi Jenis Hewan (Tetap ada, tapi dalam format chart)
    animal_counts = df[ANIMAL_COL].value_counts().head(5)
    with col5:
        st.subheader("Distribusi Jenis Hewan Terbanyak")
        st.bar_chart(animal_counts)
        # 

    st.markdown("___")

    # --- Row 3: Analisis Tren Penyakit Per Tahun (BARU & PENTING) ---
    st.subheader("Tren 5 Penyakit Terbanyak dari Tahun ke Tahun")
    
    # 1. Identifikasi 5 diagnosis teratas secara keseluruhan
    top_5_diseases = df[Y_COL].value_counts().head(5).index.tolist()
    
    # 2. Hitung jumlah kasus per tahun untuk 5 penyakit teratas tersebut
    df_trend = df[df[Y_COL].isin(top_5_diseases)].groupby(['Tahun', Y_COL]).size().reset_index(name='Jumlah Kasus')
    
    # 3. Pivot data untuk Streamlit (Tahun sebagai Index, Diagnosis sebagai Kolom)
    df_pivot = df_trend.pivot_table(index='Tahun', columns=Y_COL, values='Jumlah Kasus', fill_value=0)
    
    # Tampilkan Line Chart
    st.line_chart(df_pivot)
    # 

    st.markdown(f"""
    <p style='font-size: small; color: gray;'>
    Visualisasi di atas menunjukkan tren kasus dari 5 diagnosis terbanyak ({', '.join(top_5_diseases)}).
    Pola ini dapat mengindikasikan lonjakan kasus musiman atau tren epidemiologi.
    </p>
    """, unsafe_allow_html=True)
# Muat aset model dan data
model_pipeline, label_encoder = load_assets()
RAW_DF = load_raw_data()

# --- 5. Fungsi Utama Aplikasi Streamlit ---
def main():
    st.set_page_config(page_title="Prediksi Penyakit Hewan", layout="centered")

    st.title("Vet Diagnosa AI: Klasifikasi Penyakit Hewan")
    st.markdown("""
        **Tool Prediksi Diagnosis** menggunakan model *Machine Learning* yang dilatih dari data kasus dan gejala klinis.
    """)
    st.markdown("---")
    
    # --- Input Teks dari User ---
    input_text = st.text_area(
        "**1. Masukkan Ciri-ciri Kasus (Gejala)**", 
        placeholder="Contoh: Demam, batuk, leleran hidung, dan ada pembengkakan pada kelenjar getah bening.",
        height=150
    )

    # --- Input Jenis Hewan ---
    # Daftar Jenis Hewan yang diekstrak (Harus sesuai dengan yang ditemukan di Cell 1 notebook)
    animal_list = ['Sapi', 'Kambing', 'Kucing', 'Anjing', 'Lainnya'] 
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

            st.header("Hasil Prediksi")
            st.success(f"Diagnosis yang Diprediksi: **{predicted_diagnosis}**")
            
            st.markdown("---")
            st.info("Prediksi ini adalah output Machine Learning dan harus dikonfirmasi oleh profesional yang kompeten.")

        except Exception as e:
            st.error(f"Gagal saat prediksi.")
            st.code(f"Error detail: {e}") 

    # --- TAMPILAN TMI (INSIGHTS) ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    display_tmi(RAW_DF)

# Jalankan Aplikasi
if __name__ == "__main__":
    main()


