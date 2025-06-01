import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import onnxruntime as ort
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Konfigurasi ---
MODEL_PATH = 'models/heart_attack_model.onnx'
ENCODERS_DIR = 'models/encoders'

# Kolom-kolom fitur yang diharapkan model (harus sesuai dengan pelatihan)
EXPECTED_COLUMNS = [
    'age', 'gender', 'hypertension', 'diabetes', 'obesity',
    'waist_circumference', 'smoking_status', 'alcohol_consumption',
    'triglycerides', 'previous_heart_disease', 'medication_usage',
    'participated_in_free_screening'
]

# Kolom kategori yang perlu di-encode (nilai string seperti 'Male', 'Never Smoker')
CATEGORICAL_COLUMNS = ['gender', 'smoking_status', 'alcohol_consumption']

# Kolom numerik yang perlu di-scale (nilai continuous seperti usia, lingkar pinggang)
NUMERICAL_COLUMNS_TO_SCALE = ['age', 'waist_circumference', 'triglycerides']

# Label prediksi akhir
PREDICTION_LABELS = {
    0: "Tidak Berisiko Serangan Jantung",
    1: "Berisiko Serangan Jantung"
}

# --- Pemuatan Model dan Encoders ---
ort_session = None
scaler = None
target_label_encoder = None
le_dict = {}

try:
    ort_session = ort.InferenceSession(MODEL_PATH)
    
    # Muat LabelEncoder untuk fitur kategorikal
    with open(os.path.join(ENCODERS_DIR, 'gender_encoder.pkl'), 'rb') as f:
        le_dict['gender'] = pickle.load(f)
    with open(os.path.join(ENCODERS_DIR, 'smoking_status_encoder.pkl'), 'rb') as f:
        le_dict['smoking_status'] = pickle.load(f)
    with open(os.path.join(ENCODERS_DIR, 'alcohol_consumption_encoder.pkl'), 'rb') as f:
        le_dict['alcohol_consumption'] = pickle.load(f)
    
    # Muat LabelEncoder untuk target 
    with open(os.path.join(ENCODERS_DIR, 'label_encoder.pkl'), 'rb') as f:
        target_label_encoder = pickle.load(f)
        
    # Muat Scaler untuk fitur numerik
    with open(os.path.join(ENCODERS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
        
    print("[*] Model dan encoder berhasil dimuat!")

except Exception as e:
    print(f"[!] ERROR FATAL: Gagal memuat model atau encoder: {e}")
    print("[!] Pastikan semua file model dan encoder tersimpan dengan benar.")
    exit()

def encode_kategori(le, nilai):
    """Mengkodekan nilai kategori menggunakan LabelEncoder, menangani nilai tidak dikenal."""
    nilai_str = str(nilai)
    if nilai_str in le.classes_:
        return le.transform([nilai_str])[0]
    return 0

# --- Definisi Endpoint API ---
@app.route('/')
def home():
    feature_options = {
        'gender': ['Male', 'Female'],
        'smoking_status': ['Never Smoker', 'Current Smoker', 'Former Smoker'],
        'alcohol_consumption': ['Never', 'Rarely', 'Regularly', 'Daily'],
        'hypertension': ['0', '1'], 'diabetes': ['0', '1'], 'obesity': ['0', '1'],
        'previous_heart_disease': ['0', '1'], 'medication_usage': ['0', '1'],
        'participated_in_free_screening': ['0', '1']
    }
    return render_template('index.html', 
                           expected_features=EXPECTED_COLUMNS,
                           categorical_columns=CATEGORICAL_COLUMNS,
                           feature_options=feature_options)

@app.route('/predict', methods=['POST'])
def prediksi_serangan_jantung():
    if not request.is_json:
        return jsonify({"error": "Permintaan harus dalam format JSON"}), 400

    data = request.get_json()

    missing_keys = [key for key in EXPECTED_COLUMNS if key not in data]
    if missing_keys:
        return jsonify({
            "error": "Fitur-fitur yang diperlukan tidak ada dalam permintaan JSON.",
            "missing_features": missing_keys,
            "expected_features": EXPECTED_COLUMNS
        }), 400

    try:
        df_input = pd.DataFrame([data], columns=EXPECTED_COLUMNS)

        # Preprocessing: Encoding fitur kategorikal
        for col in CATEGORICAL_COLUMNS:
            df_input[col] = df_input[col].apply(lambda x: encode_kategori(le_dict[col], x))

        # Preprocessing: Scaling fitur numerik continuous
        for col in NUMERICAL_COLUMNS_TO_SCALE:
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
            
            # Imputasi nilai NaN dengan mean dari scaler (penting agar scaler tidak error)
            col_idx_in_scaler = list(scaler.feature_names_in_).index(col)
            df_input[col].fillna(scaler.mean_[col_idx_in_scaler], inplace=True)
            
        # Lakukan scaling hanya pada kolom yang memang perlu di-scale
        scaled_data = scaler.transform(df_input[NUMERICAL_COLUMNS_TO_SCALE])
        df_input[NUMERICAL_COLUMNS_TO_SCALE] = scaled_data

        # Pastikan urutan kolom sesuai dengan yang diharapkan oleh model ONNX
        processed_input = df_input[EXPECTED_COLUMNS].to_numpy().astype(np.float32)

        # Inferensi dengan ONNX Runtime
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        outputs = ort_session.run([output_name], {input_name: processed_input})
        probabilitas_prediksi = outputs[0][0]

        # Decode hasil prediksi: dari indeks ke label deskriptif
        idx_label_prediksi = int(np.argmax(probabilitas_prediksi))
        label_prediksi_raw = target_label_encoder.inverse_transform([idx_label_prediksi])[0]
        
        # Konversi label target asli (misal '0' atau '1' sebagai string) ke label yang lebih deskriptif
        label_final = PREDICTION_LABELS.get(int(str(label_prediksi_raw)), "Tidak Dikenal")

        # Ambil probabilitas untuk masing-masing kelas
        probabilitas_tidak_serangan = float(probabilitas_prediksi[0])
        probabilitas_serangan = float(probabilitas_prediksi[1])
        
        # Kirim Respons JSON
        return jsonify({
            "status": "success",
            "prediction_label": label_final,
            "probability_no_heart_attack": probabilitas_tidak_serangan,
            "probability_yes_heart_attack": probabilitas_serangan,
            "input_data_received": data # Mengembalikan input untuk referensi
        })

    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan saat memproses permintaan: {str(e)}"}), 500

if __name__ == '__main__':
    print("\n[+] Aplikasi Flask siap dijalankan.")
    print(f"    Akses UI: http://127.0.0.1:5000/")
    print(f"    Endpoint API: http://127.0.0.1:5000/predict (POST)")
    app.run(debug=True, host='0.0.0.0', port=5000)