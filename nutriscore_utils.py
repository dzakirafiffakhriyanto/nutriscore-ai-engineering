"""Modul pendukung proyek klasifikasi Nutri-Score (fitur baris-per-baris, tanpa kebocoran data)."""
import numpy as np  # library numerik untuk operasi array
import pandas as pd  # library tabel data
from sklearn.base import BaseEstimator, ClassifierMixin, clone  # kelas dasar estimator sklearn

# ---------------------------------------------------------------- konstanta kolom
NUTRIENT_COLS = [  # daftar kolom kandungan gizi per 100 g
    "energy_kcal", "fat_100g", "saturated_fat_100g", "carbs_100g",  # energi, lemak, lemak jenuh, karbohidrat
    "sugars_100g", "fiber_100g", "proteins_100g", "salt_100g", "sodium_100g",  # gula, serat, protein, garam, natrium
]  # akhir daftar kolom gizi
ALLERGEN_COLS = [  # daftar kolom penanda alergen (boolean)
    "contains_gluten", "contains_dairy", "contains_nuts",  # gluten, susu, kacang
    "contains_soy", "contains_eggs", "contains_fish",  # kedelai, telur, ikan
]  # akhir daftar alergen
RAW_INPUT_COLS = NUTRIENT_COLS + ["categories"] + ALLERGEN_COLS  # kolom mentah minimum yang dibutuhkan model
ENGINEERED_NUM = ["sugar_to_carb", "sat_to_fat", "macro_sum", "fiber_missing"]  # fitur rasio hasil rekayasa
POINT_NUM = [  # fitur poin pendekatan algoritma Nutri-Score
    "p_energy", "p_sugar", "p_satfat", "p_sodium", "p_fiber", "p_protein",  # poin per komponen
    "neg_points", "pos_points", "nutri_points",  # total poin negatif, positif, dan skor akhir
]  # akhir daftar poin
CATEGORY_FLAGS = [  # flag kategori produk dari teks 'categories'
    "is_beverage", "is_water", "is_cheese", "is_fat_oil",  # minuman, air, keju, lemak/minyak
    "is_fruitveg", "is_dairy", "is_snack", "is_cereal", "is_sweet",  # buah-sayur, susu, camilan, serealia, manis
]  # akhir daftar flag
CATEGORY_CAT = ["main_category"]  # kolom kategorikal kategori utama
# pola regex untuk tiap flag kategori
_FLAG_PATTERNS = {  # kamus nama flag -> pola teks
    "is_beverage": "beverage",  # minuman
    "is_water": "waters",  # air mineral
    "is_cheese": "cheese",  # keju
    "is_fat_oil": r"en:fats|oils",  # lemak dan minyak
    "is_fruitveg": "fruits|vegetables|legumes",  # buah, sayur, kacang-kacangan
    "is_dairy": "dairies|milk",  # produk susu
    "is_snack": "snacks",  # camilan
    "is_cereal": "cereals",  # serealia
    "is_sweet": "sweet|cocoa|biscuits|chocolate",  # produk manis
}  # akhir kamus pola


# ---------------------------------------------------------------- normalisasi tipe data
def normalize_types(df):
    """Ubah kolom gizi/NOVA menjadi numerik dan kolom alergen menjadi 0/1 (database SQLite menyimpan semuanya sebagai teks)."""
    out = df.copy()  # salin agar data asli tidak berubah
    for col in NUTRIENT_COLS + ["nova_group"]:  # kolom yang seharusnya numerik
        if col in out.columns:  # hanya jika kolom ada
            out[col] = pd.to_numeric(out[col], errors="coerce")  # teks tidak valid menjadi NaN
    for col in ALLERGEN_COLS:  # kolom alergen
        if col in out.columns:  # hanya jika kolom ada
            out[col] = out[col].astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes", "t"]).astype(int)  # teks/boolean -> 0/1
    return out  # kembalikan data bertipe benar


# ---------------------------------------------------------------- pembersihan nilai tidak wajar
def clean_nutrients(df):
    """Ubah nilai gizi yang mustahil secara fisik menjadi NaN (aturan baris-per-baris)."""
    out = df.copy()  # salin agar data asli tidak berubah
    for col in NUTRIENT_COLS[1:]:  # semua kolom gram per 100 g (selain energi)
        out[col] = pd.to_numeric(out[col], errors="coerce")  # paksa menjadi numerik
        out.loc[(out[col] > 100) | (out[col] < 0), col] = np.nan  # >100 g atau negatif per 100 g mustahil
    out["energy_kcal"] = pd.to_numeric(out["energy_kcal"], errors="coerce")  # energi numerik
    out.loc[(out["energy_kcal"] > 900) | (out["energy_kcal"] < 0), "energy_kcal"] = np.nan  # maksimum fisik lemak murni = 900 kcal
    out.loc[out["saturated_fat_100g"] > out["fat_100g"] + 1e-9, "saturated_fat_100g"] = np.nan  # lemak jenuh tak boleh > lemak total
    out.loc[out["sugars_100g"] > out["carbs_100g"] + 1e-9, "sugars_100g"] = np.nan  # gula tak boleh > karbohidrat total
    return out  # kembalikan data bersih


# ---------------------------------------------------------------- poin Nutri-Score (pendekatan)
def _points(series, thresholds):
    """Hitung poin = jumlah ambang yang dilewati nilai; NaN dipertahankan."""
    values = series.astype(float)  # pastikan tipe float
    pts = pd.Series(0.0, index=values.index)  # mulai dari 0 poin
    for t in thresholds:  # untuk setiap ambang
        pts = pts + (values > t).astype(float)  # tambah 1 poin bila nilai melewati ambang
    pts[values.isna()] = np.nan  # nilai kosong -> poin kosong
    return pts  # kembalikan poin


def prepare_features(df):
    """Bersihkan nilai mustahil lalu tambahkan fitur turunan. Tidak ada parameter yang dipelajari dari data."""
    out = clean_nutrients(normalize_types(df))  # samakan tipe data lalu bersihkan nilai tidak wajar
    cat = out["categories"].fillna("").astype(str).str.lower()  # teks kategori huruf kecil
    for name, pattern in _FLAG_PATTERNS.items():  # untuk tiap flag kategori
        out[name] = cat.str.contains(pattern, regex=True).astype(int)  # 1 bila pola ditemukan
    first = cat.str.split(",").str[0].str.strip()  # kategori pertama (paling umum)
    out["main_category"] = first.where(first != "", "unknown")  # kategori utama; kosong -> unknown
    out["sugar_to_carb"] = out["sugars_100g"] / (out["carbs_100g"] + 1e-3)  # proporsi gula dalam karbohidrat
    out["sat_to_fat"] = out["saturated_fat_100g"] / (out["fat_100g"] + 1e-3)  # proporsi lemak jenuh dalam lemak
    out["macro_sum"] = out[["fat_100g", "carbs_100g", "proteins_100g"]].sum(axis=1, skipna=False)  # total makronutrien per 100 g
    out["fiber_missing"] = out["fiber_100g"].isna().astype(int)  # penanda serat tidak dilaporkan
    out["p_energy"] = _points(out["energy_kcal"] * 4.184, [335 * i for i in range(1, 11)])  # poin energi (kJ)
    out["p_sugar"] = _points(out["sugars_100g"], [4.5 * i for i in range(1, 11)])  # poin gula
    out["p_satfat"] = _points(out["saturated_fat_100g"], list(range(1, 11)))  # poin lemak jenuh
    out["p_sodium"] = _points(out["sodium_100g"] * 1000, [90 * i for i in range(1, 11)])  # poin natrium (mg)
    out["p_fiber"] = _points(out["fiber_100g"], [0.9, 1.9, 2.8, 3.7, 4.7])  # poin serat (positif)
    out["p_protein"] = _points(out["proteins_100g"], [1.6, 3.2, 4.8, 6.4, 8.0])  # poin protein (positif)
    out["neg_points"] = out[["p_energy", "p_sugar", "p_satfat", "p_sodium"]].sum(axis=1, skipna=False)  # total poin negatif
    out["pos_points"] = out[["p_fiber", "p_protein"]].sum(axis=1, skipna=True)  # total poin positif (serat kosong dianggap 0)
    out["nutri_points"] = out["neg_points"] - out["pos_points"]  # skor gizi pendekatan (makin kecil makin sehat)
    return out  # kembalikan data dengan fitur lengkap (alergen sudah 0/1 dari normalize_types)


def rule_based_grade(df):
    """Baseline non-ML: ubah skor poin pendekatan menjadi grade A-E dengan ambang pendekatan; bukan kalkulator resmi penuh."""
    score = df["nutri_points"].to_numpy(dtype=float)  # skor gizi
    is_bev = df["is_beverage"].to_numpy(dtype=int) == 1  # penanda minuman
    food = np.select([score <= -1, score <= 2, score <= 10, score <= 18], [0, 1, 2, 3], default=4)  # ambang makanan umum
    bev = np.select([score <= 1, score <= 5, score <= 9], [1, 2, 3], default=4)  # ambang minuman (A hanya untuk air)
    bev = np.where(df["is_water"].to_numpy(dtype=int) == 1, 0, bev)  # air mineral -> grade A
    pred = np.where(is_bev, bev, food)  # pilih ambang sesuai jenis produk
    pred = np.where(np.isnan(score), 2, pred)  # skor kosong -> tebak kelas tengah (C)
    return pred.astype(int)  # kembalikan indeks grade 0..4


# ---------------------------------------------------------------- klasifier ordinal
class OrdinalClassifier(ClassifierMixin, BaseEstimator):
    """Klasifier ordinal pendekatan Frank & Hall: K-1 model biner untuk P(y > k)."""

    def __init__(self, base_estimator=None):  # konstruktor menerima estimator dasar
        self.base_estimator = base_estimator  # simpan estimator dasar

    def fit(self, X, y):  # latih K-1 model biner
        y = np.asarray(y)  # pastikan array numpy
        self.classes_ = np.sort(np.unique(y))  # daftar kelas terurut
        self.models_ = []  # tempat menyimpan model biner
        for k in self.classes_[:-1]:  # untuk setiap ambang kelas
            model = clone(self.base_estimator)  # salin estimator dasar yang belum dilatih
            model.fit(X, (y > k).astype(int))  # latih untuk memprediksi apakah y > k
            self.models_.append(model)  # simpan model
        return self  # kembalikan diri sendiri

    def predict_proba(self, X):  # hitung probabilitas tiap kelas
        gt = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])  # matriks P(y > k)
        n_cls = len(self.classes_)  # jumlah kelas
        probs = np.zeros((gt.shape[0], n_cls))  # tempat probabilitas kelas
        probs[:, 0] = 1 - gt[:, 0]  # P(y = kelas pertama)
        for i in range(1, n_cls - 1):  # kelas-kelas tengah
            probs[:, i] = gt[:, i - 1] - gt[:, i]  # P(y > i-1) - P(y > i)
        probs[:, -1] = gt[:, -1]  # P(y = kelas terakhir)
        probs = np.clip(probs, 1e-9, None)  # hindari probabilitas negatif akibat ketidakkonsistenan
        return probs / probs.sum(axis=1, keepdims=True)  # normalisasi agar jumlah = 1

    def predict(self, X):  # prediksi kelas
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]  # kelas dengan probabilitas terbesar
