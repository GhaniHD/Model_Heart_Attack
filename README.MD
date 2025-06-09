# Prediksi Serangan Jantung (Heart Attack Prediction)

Model ini dikembangkan untuk memprediksi kemungkinan serangan jantung berdasarkan beberapa fitur kesehatan dan gaya hidup. Model ini telah melalui proses *preprocessing* data yang komprehensif, pelatihan menggunakan Neural Network (TensorFlow/Keras), dan diekspor ke format ONNX untuk inferensi yang lebih efisien di lingkungan produksi.

---

## 1. Untuk Tim *Frontend*

Tim *frontend* perlu memahami data yang akan dikirimkan ke *backend* untuk mendapatkan prediksi. Pastikan formulir input pengguna sesuai dengan format data yang dibutuhkan model.

### Input Data yang Dibutuhkan

Model membutuhkan data dalam format spesifik. Berikut adalah fitur-fitur yang diharapkan beserta tipe datanya:

| Fitur                            | Tipe Data Asli       | Contoh                                   | Keterangan                                              | Cara Input                                         |
| :------------------------------- | :------------------- | :--------------------------------------- | :------------------------------------------------------ | :------------------------------------------------- |
| `age`                            | Numerik (Integer)    | `55`                                     | Usia pasien.                                            | Input angka                                        |
| `gender`                         | Kategorikal (String) | `Male`, `Female`                         | Jenis kelamin pasien.                                   | Dropdown: `Male`, `Female`                         |
| `hypertension`                   | Numerik (Binary)     | `0`, `1`                                 | Riwayat hipertensi (0: Tidak, 1: Ya).                   | Checkbox / Toggle                                  |
| `diabetes`                       | Numerik (Binary)     | `0`, `1`                                 | Riwayat diabetes (0: Tidak, 1: Ya).                     | Checkbox / Toggle                                  |
| `obesity`                        | Numerik (Binary)     | `0`, `1`                                 | Riwayat obesitas (0: Tidak, 1: Ya).                     | Checkbox / Toggle                                  |
| `waist_circumference`            | Numerik (Float)      | `90.5`                                   | Lingkar pinggang (dalam cm).                            | Input angka desimal                                |
| `smoking_status`                 | Kategorikal (String) | `Non-smoker`, `Former Smoker`, `Regular` | Status merokok pasien.                                  | Dropdown: `Non-smoker`, `Former Smoker`, `Regular` |
| `alcohol_consumption`            | Kategorikal (String) | `Never`, `Moderate`, `Heavy`             | Tingkat konsumsi alkohol pasien.                        | Dropdown: `Never`, `Moderate`, `Heavy`             |
| `triglycerides`                  | Numerik (Float)      | `150.2`                                  | Tingkat trigliserida dalam darah (mg/dL).               | Input angka desimal                                |
| `previous_heart_disease`         | Numerik (Binary)     | `0`, `1`                                 | Riwayat penyakit jantung sebelumnya (0: Tidak, 1: Ya).  | Checkbox / Toggle                                  |
| `medication_usage`               | Numerik (Binary)     | `0`, `1`                                 | Penggunaan obat-obatan (0: Tidak, 1: Ya).               | Checkbox / Toggle                                  |
| `participated_in_free_screening` | Numerik (Binary)     | `0`, `1`                                 | Partisipasi dalam *screening* gratis (0: Tidak, 1: Ya). | Checkbox / Toggle                                  |


---

### Output Prediksi

*Backend* akan mengembalikan hasil prediksi yang bisa berupa:
* **Probabilitas:** Angka antara 0 dan 1 untuk setiap kelas (misalnya, probabilitas tidak serangan jantung, probabilitas serangan jantung).
* **Label Prediksi:** `Yes` (kemungkinan serangan jantung tinggi) atau `No` (kemungkinan serangan jantung rendah).

Disarankan untuk menampilkan probabilitas dan label prediksi untuk memberikan informasi yang lebih lengkap kepada pengguna.

---

## 2. Untuk Tim *Backend*

Tim *backend* bertanggung jawab untuk menerima data dari *frontend*, melakukan *preprocessing* sesuai dengan pipeline model, menjalankan inferensi menggunakan model ONNX, dan mengembalikan hasilnya.

### Struktur Proyek Model

Model dan objek yang diperlukan untuk *preprocessing* disimpan dalam direktori `models/`.

```
models/
├── heart_attack_model.h5         # Model Keras asli (untuk referensi/pengembangan)
├── heart_attack_model.onnx       # Model dalam format ONNX (digunakan untuk inferensi)
└── encoders/
    ├── gender_encoder.pkl        # LabelEncoder untuk 'gender'
    ├── smoking_status_encoder.pkl # LabelEncoder untuk 'smoking_status'
    ├── alcohol_consumption_encoder.pkl # LabelEncoder untuk 'alcohol_consumption'
    ├── label_encoder.pkl         # LabelEncoder untuk variabel target 'heart_attack'
    └── scaler.pkl                # StandardScaler untuk fitur numerik
```

---

### Dependensi Python

Pastikan lingkungan *backend* Anda memiliki dependensi berikut terinstal:
* `pandas`
* `numpy`
* `scikit-learn`
* `tensorflow` (jika Anda perlu memuat model .h5 atau untuk referensi)
* `onnxruntime`
* `tf2onnx` (jika Anda perlu mengkonversi ulang model)

Anda bisa menginstal ini menggunakan `pip`:
```bash
pip install pandas numpy scikit-learn tensorflow onnxruntime tf2onnx
```

---

### Alur Kerja Inferensi di *Backend*

Berikut adalah langkah-langkah yang harus dilakukan oleh *backend* untuk memproses permintaan prediksi:

1.  **Muat Model dan Objek Preprocessing:**
    * Muat model ONNX: `onnxruntime.InferenceSession('models/heart_attack_model.onnx')`
    * Muat `LabelEncoder` untuk kolom kategorikal (`gender`, `smoking_status`, `alcohol_consumption`) dari `models/encoders/`.
    * Muat `StandardScaler` dari `models/encoders/scaler.pkl`.
    * Muat `LabelEncoder` untuk variabel target (`heart_attack`) dari `models/encoders/label_encoder.pkl` untuk decoding hasil prediksi.

    **Contoh loading:**
    ```python
    import pickle
    import onnxruntime as ort

    # Load ONNX model
    ort_session = ort.InferenceSession("models/heart_attack_model.onnx")

    # Load encoders and scaler
    with open('models/encoders/gender_encoder.pkl', 'rb') as f:
        gender_encoder = pickle.load(f)
    with open('models/encoders/smoking_status_encoder.pkl', 'rb') as f:
        smoking_status_encoder = pickle.load(f)
    with open('models/encoders/alcohol_consumption_encoder.pkl', 'rb') as f:
        alcohol_consumption_encoder = pickle.load(f)
    with open('models/encoders/label_encoder.pkl', 'rb') as f:
        target_label_encoder = pickle.load(f)
    with open('models/encoders/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Dictionary untuk menyimpan semua label encoder
    le_dict = {
        'gender': gender_encoder,
        'smoking_status': smoking_status_encoder,
        'alcohol_consumption': alcohol_consumption_encoder
    }
    ```

2.  **Terima Data Mentah dari *Frontend***:
    Data akan diterima biasanya dalam format JSON dari permintaan API.

    **Contoh data yang diterima:**
    ```python
    new_data = {
        'age': 55,
        'gender': 'Male',
        'hypertension': 1,
        'diabetes': 0,
        'obesity': 0,
        'waist_circumference': 90,
        'smoking_status': 'Regular',
        'alcohol_consumption': 'Moderate',
        'triglycerides': 150,
        'previous_heart_disease': 0,
        'medication_usage': 0,
        'participated_in_free_screening': 1,
    }
    ```

3.  **Preprocessing Data Input:**
    * **Buat DataFrame:** Konversi data input menjadi Pandas DataFrame.
    * **Encoding Kategorikal:** Gunakan `LabelEncoder` yang telah dimuat untuk mentransformasi kolom `gender`, `smoking_status`, dan `alcohol_consumption`. Pastikan penanganan `unknown_value` jika ada nilai yang tidak ada dalam kelas yang dikenal oleh encoder.
        ```python
        def safe_label_encode(le, value):
            # Cek apakah nilai ada di kelas encoder, jika tidak, kembalikan 0 (atau nilai default lain)
            if value in le.classes_:
                return le.transform([value])[0]
            else:
                # Ini bisa disesuaikan, misalnya, mengangkat error atau menggunakan nilai default.
                # Untuk tujuan prediksi, 0 seringkali merupakan pilihan aman jika tidak ada di training data.
                return 0 # Atau le.transform([le.classes_[0]]) jika ingin default ke kelas pertama

        df_new = pd.DataFrame([new_data]) # Pastikan ini adalah list of dicts

        categorical_columns = ['gender', 'smoking_status', 'alcohol_consumption']
        for col in categorical_columns:
            df_new[col] = df_new[col].apply(lambda x: safe_label_encode(le_dict[col], x))
        ```
    * **Scaling Numerik:** Gunakan `StandardScaler` yang telah dimuat untuk mentransformasi kolom `age`, `waist_circumference`, dan `triglycerides`.
        ```python
        numerical_columns = ['age', 'waist_circumference', 'triglycerides']
        df_new[numerical_columns] = scaler.transform(df_new[numerical_columns])
        ```
    * **Urutan Kolom:** Pastikan urutan kolom pada DataFrame input sama persis dengan urutan kolom yang digunakan saat melatih model (`X_train.columns`).

        **Daftar kolom yang diharapkan oleh model:**
        ```python
        expected_columns = [
            'age', 'gender', 'hypertension', 'diabetes', 'obesity',
            'waist_circumference', 'smoking_status', 'alcohol_consumption',
            'triglycerides', 'previous_heart_disease', 'medication_usage',
            'participated_in_free_screening'
        ]
        input_data_processed = df_new[expected_columns].to_numpy().astype(np.float32)
        ```

4.  **Jalankan Inferensi dengan ONNX Runtime:**
    ```python
    import numpy as np

    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: input_data_processed})
    y_pred_prob = outputs[0] # Probabilitas output
    ```

5.  **Proses Output Prediksi:**
    * **Dapatkan label prediksi:** `y_pred_label = np.argmax(y_pred_prob, axis=1)`
    * **Dekode label ke teks:** Gunakan `target_label_encoder.inverse_transform(y_pred_label)` untuk mengubah kembali label angka (0 atau 1) menjadi `No` atau `Yes`.
    * **Konversi ke format *friendly*:** Ubah `0` menjadi `No` dan `1` menjadi `Yes` secara eksplisit jika diinginkan.

    ```python
    y_pred_decoded = target_label_encoder.inverse_transform(y_pred_label)
    mapping = {0: "No", 1: "Yes"}
    final_prediction = [mapping[label] for label in y_pred_label] # Ini adalah prediksi Yes/No
    ```

6.  **Kirim Hasil Prediksi ke *Frontend***:
    Kembalikan probabilitas dan label prediksi yang telah didekode ke *frontend* dalam format JSON.

    **Contoh output JSON:**
    ```json
    {
        "prediction_label": "Yes",
        "prediction_probability": [0.1, 0.9] // Probabilitas untuk [No, Yes]
    }
    ```

---

### Catatan Penting

* **Konsistensi Data:** Pastikan data yang dikirim dari *frontend* dan diproses di *backend* konsisten dengan data yang digunakan untuk melatih model. Perbedaan dalam format atau nilai bisa menyebabkan hasil prediksi yang tidak akurat.
* **Penanganan Error:** Implementasikan penanganan error yang robust untuk kasus di mana data input tidak valid atau ada masalah saat memuat model/encoder.
* **Versi Dependensi:** Usahakan untuk menggunakan versi library yang sama antara lingkungan pelatihan model dan lingkungan produksi untuk menghindari masalah kompatibilitas.
* **Optimasi ONNX:** ONNX Runtime sangat efisien. Anda bisa memanfaatkan *session options* untuk optimasi lebih lanjut jika diperlukan.

---