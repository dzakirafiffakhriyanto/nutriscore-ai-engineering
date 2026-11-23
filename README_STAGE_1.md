# NutriScore — Stage 1 Model Artifact Package

Package ini berisi model yang sudah dilatih ulang dan diuji pemuatan ulangnya.

## Isi package

- `nutriscore_utils.py` — fungsi preprocessing/feature engineering dan kelas `OrdinalClassifier` yang dipakai pipeline.
- `outputs_nutriscore/nutriscore_pipeline.joblib` — pipeline lengkap: feature engineering, preprocessing, dan model Ordinal-HGB.
- `outputs_nutriscore/metadata.json` — identitas model, parameter, feature contract, versi library, dan metrik hasil rebuild.
- `outputs_nutriscore/artifact_checksums.json` — SHA-256 untuk file artefak.
- `requirements.txt` — versi dependency inti yang sesuai untuk artefak di package ini.
- `sample_input.json` — contoh struktur input untuk tahap API berikutnya.

## Cara memuat model

Jalankan dari folder package ini setelah menginstall dependency:

```bash
pip install -r requirements.txt
python -c "import joblib; m=joblib.load('outputs_nutriscore/nutriscore_pipeline.joblib'); print(type(m).__name__)"
```

Pastikan `nutriscore_utils.py` berada di root folder yang sama dengan folder `outputs_nutriscore/`, karena model menggunakan `nu.OrdinalClassifier` dari modul tersebut.

## Validasi yang sudah dilakukan

- Model terpilih: `Ordinal-HGB (tuned)`.
- Model berhasil disimpan dan dimuat ulang.
- Prediksi dan probabilitas dari model sebelum dan sesudah reload identik pada sampel uji.
- Train/test tidak memiliki grup produk-merek yang beririsan.

## Catatan reproduksibilitas penting

Notebook yang diunggah sebelumnya menyimpan output eksekusi dengan scikit-learn `1.6.1`. Package ini dibangun ulang pada environment aktif scikit-learn `1.8.0`, sehingga ukuran split dan metrik test dapat berbeda. Metrik package ini dicatat apa adanya di `outputs_nutriscore/metadata.json`. Gunakan dependency di `requirements.txt` untuk memuat artefak ini.

Di notebook versi `NutriScore_Supervised_Learning_Stage1_Ready.ipynb`, cell ekspor Tahap 1 mengambil nilai hyperparameter penuh dari `search_hgb.best_params_` pada runtime notebook yang sedang digunakan. Dengan demikian, jika ingin mempertahankan lingkungan eksekusi notebook sebelumnya, jalankan semua cell di runtime tersebut dan ekspor dari sana.

## Batasan model

Model memperkirakan label berdasarkan pola dataset proyek. Model ini bukan implementasi lengkap kalkulator resmi Nutri-Score. Skor kelas (`predict_proba`) juga belum dinyatakan terkalibrasi.
