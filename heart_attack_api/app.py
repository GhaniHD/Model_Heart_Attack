import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import onnxruntime as ort
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Path ke model dan encoders
MODEL_PATH = 'models/heart_attack_model.onnx'
ENCODERS_DIR = 'models/encoders'

# Definisikan kolom-kolom yang diharapkan oleh model
EXPECTED_COLUMNS = [
    'age', 'gender', 'hypertension', 'diabetes', 'obesity',
    'waist_circumference', 'smoking_status', 'alcohol_consumption',
    'triglycerides', 'previous_heart_disease', 'medication_usage',
    'participated_in_free_screening'
]

# Muat model dan encoders saat aplikasi dimulai
try:
    ort_session = ort.InferenceSession(MODEL_PATH)

    with open(os.path.join(ENCODERS_DIR, 'gender_encoder.pkl'), 'rb') as f:
        gender_encoder = pickle.load(f)
    with open(os.path.join(ENCODERS_DIR, 'smoking_status_encoder.pkl'), 'rb') as f:
        smoking_status_encoder = pickle.load(f)
    with open(os.path.join(ENCODERS_DIR, 'alcohol_consumption_encoder.pkl'), 'rb') as f:
        alcohol_consumption_encoder = pickle.load(f)
    with open(os.path.join(ENCODERS_DIR, 'label_encoder.pkl'), 'rb') as f:
        target_label_encoder = pickle.load(f)
    with open(os.path.join(ENCODERS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    le_dict = {
        'gender': gender_encoder,
        'smoking_status': smoking_status_encoder,
        'alcohol_consumption': alcohol_consumption_encoder
    }

    print("Model dan encoders berhasil dimuat!")

except Exception as e:
    print(f"Error saat memuat model atau encoders: {e}")
    exit()

def safe_label_encode(le, value):
    """Mengkodekan nilai kategorikal dengan LabelEncoder. Menangani nilai tidak dikenal."""
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        return 0

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Permintaan harus dalam format JSON"}), 400

    data = request.get_json()

    missing_keys = [key for key in EXPECTED_COLUMNS if key not in data]
    if missing_keys:
        return jsonify({"error": f"Kunci yang hilang dalam JSON: {', '.join(missing_keys)}"}), 400

    try:
        # Konversi data input ke DataFrame
        df_input = pd.DataFrame([data])

        # Preprocessing: Encoding fitur kategorikal
        categorical_columns = ['gender', 'smoking_status', 'alcohol_consumption']
        for col in categorical_columns:
            if col in df_input.columns and col in le_dict:
                df_input[col] = df_input[col].apply(lambda x: safe_label_encode(le_dict[col], x))
            else:
                # Jika kolom kategorikal tidak ada atau encoder tidak dimuat, set ke 0
                df_input[col] = 0

        # Preprocessing: Scaling fitur numerikal
        numerical_columns = ['age', 'waist_circumference', 'triglycerides']

        # Pastikan hanya kolom numerik yang ada di input yang akan diskalakan
        # dan buat salinan untuk menghindari SettingWithCopyWarning
        df_numerical = df_input[numerical_columns].copy()

        # Skala semua kolom numerik sekaligus
        scaled_data = scaler.transform(df_numerical)

        # Masukkan kembali data yang sudah diskalakan ke DataFrame input utama
        df_input[numerical_columns] = scaled_data

        # Pastikan urutan kolom sesuai dengan yang diharapkan oleh model
        # Gunakan EXPECTED_COLUMNS untuk reindex DataFrame
        processed_input = df_input[EXPECTED_COLUMNS].to_numpy().astype(np.float32)

        # Lakukan inferensi
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: processed_input})
        y_pred_proba = outputs[0][0]

        # Decode hasil prediksi
        y_pred_label_idx = int(np.argmax(y_pred_proba))
        predicted_label_raw = target_label_encoder.inverse_transform([y_pred_label_idx])[0]

        if str(predicted_label_raw) == '0':
            predicted_label = "Tidak"
        elif str(predicted_label_raw) == '1':
            predicted_label = "Ya"
        else:
            predicted_label = str(predicted_label_raw) 

        probability_no_attack = float(y_pred_proba[0])
        probability_yes_attack = float(y_pred_proba[1])


        return jsonify({
            "prediction_label": predicted_label,
            "probability_no_heart_attack": float(probability_no_attack),
            "probability_yes_heart_attack": float(probability_yes_attack)
        })

    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan saat memproses permintaan: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)