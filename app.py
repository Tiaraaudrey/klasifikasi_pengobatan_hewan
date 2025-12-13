# app.py
import joblib
from flask import Flask, request, jsonify
import numpy as np
import os # import os untuk pengecekan file

# --- 1. MUAT ASET MODEL ---
try:
    model_pipeline = joblib.load('model_assets/ai_diagnosa_pipeline.pkl')
    label_encoder = joblib.load('model_assets/label_encoder.pkl')
    print("Pipeline model dan Label Encoder berhasil dimuat.")
except Exception as e:
    print(f"Error memuat aset model: {e}")
    exit() 

app = Flask(__name__)

# --- 2. DEFINISI ENDPOINT API ---
@app.route('/predict', methods=['POST'])
def predict_diagnosis():
    data = request.get_json(silent=True)
    if not data or 'ciri_kasus' not in data:
        return jsonify({"error": "Input JSON tidak valid. Diperlukan field 'ciri_kasus'."}), 400

    input_text = data['ciri_kasus']

    try:
        prediction_encoded = model_pipeline.predict([input_text])[0]
        
        predicted_diagnosis = label_encoder.inverse_transform([prediction_encoded])[0]

    except Exception as e:
        return jsonify({"error": f"Gagal saat prediksi: {e}"}), 500

    return jsonify({
        "status": "success",
        "input_ciri_kasus": input_text,
        "predicted_diagnosis": predicted_diagnosis,
    })

# --- 3. JALANKAN APLIKASI ---
if __name__ == '__main__':
    # host='0.0.0.0' agar API bisa diakses dari luar container Docker
    app.run(debug=False, host='0.0.0.0', port=5000)