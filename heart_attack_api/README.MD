

---

# API Prediksi Risiko Serangan Jantung

API ini menyediakan layanan prediksi risiko serangan jantung berdasarkan data pasien menggunakan model *machine learning*.

---

## Untuk Tim Backend

### Cara Menjalankan Aplikasi

1.  **Pastikan Struktur Folder:**
    ```
    your_project_folder/
    ├── app.py
    ├── models/
    │   ├── heart_attack_model.onnx
    │   └── encoders/
    │       ├── gender_encoder.pkl
    │       ├── smoking_status_encoder.pkl
    │       ├── alcohol_consumption_encoder.pkl
    │       ├── label_encoder.pkl
    │       └── scaler.pkl
    └── templates/
        └── index.html
    ```
    Pastikan semua file `.onnx` dan `.pkl` ada di lokasi yang benar.

2.  **Instal Dependensi:**
    ```bash
    pip install flask pandas numpy onnxruntime scikit-learn flask-cors
    ```

3.  **Jalankan Server:**
    ```bash
    python app.py
    ```
    Aplikasi akan berjalan di `http://0.0.0.0:5000/`.

### Detail Endpoint

* **`GET /`**:
    * Mengembalikan halaman web HTML sederhana (`index.html`) untuk *input* data dan pengujian manual.

* **`POST /predict`**:
    * **Deskripsi:** Menerima data pasien dan memprediksi risiko serangan jantung.
    * **`Content-Type`**: `application/json`
    * **Request Body (JSON):**
        ```json
        {
            "age": 55,
            "gender": "Female",
            "hypertension": 1,
            "diabetes": 0,
            "obesity": 1,
            "waist_circumference": 95,
            "smoking_status": "Never Smoker",
            "alcohol_consumption": "Rarely",
            "triglycerides": 180,
            "previous_heart_disease": 0,
            "medication_usage": 1,
            "participated_in_free_screening": 0
        }
        ```
        **Catatan:** Semua *key* di atas **wajib** ada.

    * **Response Body (JSON Sukses):**
        ```json
        {
            "prediction_label": "No",  
            "probability_no_heart_attack": 0.85,
            "probability_yes_heart_attack": 0.15
        }
        ```
        `prediction_label` akan berupa "Yes" atau "No".

    * **Response Body (JSON Error):**
        ```json
        {
            "error": "Pesan kesalahan detail..."
        }
        ```
        Status HTTP akan bervariasi (misal: 400 Bad Request, 500 Internal Server Error).

---

## Untuk Tim Frontend

### Integrasi dengan API

* **Dasar:** Anda akan melakukan permintaan **HTTP POST** ke `http://127.0.0.1:5000/predict`.
* **Header:** Pastikan `Content-Type` diatur ke `application/json`.
* **Data:** Kirim *body* JSON yang berisi semua 12 parameter input seperti contoh di atas.
    * Nilai numerik (`age`, `waist_circumference`, `triglycerides`) harus berupa angka.
    * Nilai biner (`hypertension`, `diabetes`, `obesity`, `previous_heart_disease`, `medication_usage`, `participated_in_free_screening`) harus `0` atau `1`.
    * Nilai kategorikal (`gender`, `smoking_status`, `alcohol_consumption`) harus berupa string sesuai daftar yang diizinkan (lihat `app.py` atau contoh *request body*).
* **CORS:** Aplikasi Flask sudah dikonfigurasi dengan **CORS enabled** (`flask-cors`) sehingga tidak akan ada masalah keamanan *cross-origin* saat integrasi dari *frontend* lokal Anda.

### Contoh Implementasi (JavaScript - Fetch API)

```javascript
async function getPrediction(data) {
    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (response.ok) { // Status kode 200-299
            console.log('Prediksi Sukses:', result);
            return result; // { "prediction_label": "No", "probability_no_heart_attack": 0.85, ... }
        } else {
            console.error('Error dari API:', result.error);
            throw new Error(result.error || 'Unknown API Error');
        }
    } catch (error) {
        console.error('Gagal terhubung ke API atau error jaringan:', error);
        throw error;
    }
}

// Contoh penggunaan:
const patientData = {
    "age": 55, "gender": "Female", "hypertension": 1, "diabetes": 0, "obesity": 1,
    "waist_circumference": 95, "smoking_status": "Never Smoker", "alcohol_consumption": "Rarely",
    "triglycerides": 180, "previous_heart_disease": 0, "medication_usage": 1,
    "participated_in_free_screening": 0
};

// getPrediction(patientData)
//     .then(prediction => {
//         // Tampilkan hasil prediksi di UI
//         console.log(prediction);
//     })
//     .catch(error => {
//         // Tangani error di UI
//         console.error('Error saat memproses prediksi:', error);
//     });
```

---
