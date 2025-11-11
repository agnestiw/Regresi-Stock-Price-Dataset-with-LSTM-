import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import warnings

# Menonaktifkan peringatan yang mungkin muncul (opsional)
warnings.filterwarnings('ignore')

# --- 1. Fungsi Evaluasi ---
def evaluasi_model(model_name, y_test, y_pred):
    """
    Fungsi untuk menghitung dan menampilkan 5 metrik evaluasi.
    """
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- Hasil Evaluasi Model: {model_name} ---")
    print(f"1. Mean Absolute Error (MAE):     {mae:.3f}")
    print(f"2. Mean Abs. Pct. Error (MAPE): {mape:.3%}") # Ditampilkan sebagai persentase
    print(f"3. Mean Squared Error (MSE):    {mse:.3f}")
    print(f"4. Root MSE (RMSE):             {rmse:.3f}")
    print(f"5. R-squared (R²):              {r2:.3f}")
    
    # Mengembalikan metrik jika perlu disimpan
    return {'mae': mae, 'mape': mape, 'mse': mse, 'rmse': rmse, 'r2': r2}

# --- 2. Muat dan Persiapan Data ---
print("=== 1. Memuat dan Memproses Data ===")

# Pastikan file 'Stock_Price_Dataset.csv' ada di folder yang sama
file_path = 'Stock_Price_Dataset.csv'

try:
    df = pd.read_csv(file_path)
    print("...Data berhasil dimuat.")
except FileNotFoundError:
    print(f"ERROR: File not found at {file_path}")
    print("Pastikan file 'Stock_Price_Dataset.csv' berada di folder yang sama dengan script ini.")
    exit() # Keluar dari script jika file tidak ditemukan

# Tampilkan 5 baris pertama
print("\nData Awal:")
print(df.head())

# Buat salinan dataframe
df_processed = df.copy()

# 1. Ubah 'Date' menjadi angka (ordinal)
df_processed['Date'] = pd.to_datetime(df_processed['Date'])
df_processed['Date_ordinal'] = df_processed['Date'].apply(lambda x: x.toordinal())

# 2. Ubah kolom kategori ('Stock', 'Risk') menjadi dummy variables
df_processed = pd.get_dummies(df_processed, columns=['Stock', 'Risk'], drop_first=True)

# Hapus kolom asli yang tidak lagi diperlukan (Date)
df_processed = df_processed.drop('Date', axis=1)

print("\nData Setelah Diproses:")
print(df_processed.head())

# --- 3. Tentukan Fitur (X) dan Target (y) ---
y = df_processed['Price']
X = df_processed.drop('Price', axis=1)

print(f"\nBentuk Fitur (X): {X.shape}")
print(f"Bentuk Target (y): {y.shape}")

# --- 4. Pisahkan Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Jumlah data training: {X_train.shape[0]}")
print(f"Jumlah data testing: {X_test.shape[0]}")

# --- 5. Model 1: AdaBoost (Training & Evaluasi) ---
print("\n=== 2. MEMULAI TRAINING MODEL 1: AdaBoost ===")
model_ada = AdaBoostRegressor(random_state=42)
model_ada.fit(X_train, y_train)
print("...Training AdaBoost Selesai.")
y_pred_ada = model_ada.predict(X_test)
evaluasi_model("AdaBoost Regressor", y_test, y_pred_ada)

# --- 6. Model 2: Gradient Boosting (Training & Evaluasi) ---
print("\n=== 3. MEMULAI TRAINING MODEL 2: Gradient Boosting ===")
model_gb = GradientBoostingRegressor(random_state=42)
model_gb.fit(X_train, y_train)
print("...Training Gradient Boosting Selesai.")
y_pred_gb = model_gb.predict(X_test)
evaluasi_model("Gradient Boosting Regressor", y_test, y_pred_gb)

# --- 7. Visualisasi ---
print("\n=== 4. Membuat Visualisasi Plot ===")
print("...Menampilkan Plot 1: Perbandingan Harga.")

# Plot 1: Perbandingan Harga Aktual vs Prediksi
plt.figure(figsize=(12, 6))
plt.scatter(X_test['Date_ordinal'], y_test, alpha=0.6, color='blue', label='Harga Aktual (Data Tes)')
sorted_indices = X_test['Date_ordinal'].argsort()
plt.scatter(X_test['Date_ordinal'].iloc[sorted_indices], y_pred_gb[sorted_indices], 
           alpha=0.6, color='red', label='Harga Prediksi (Gradient Boosting)')
plt.title('Perbandingan Harga Aktual vs Prediksi Model (Gradient Boosting)')
plt.xlabel('Hari (Ordinal)')
plt.ylabel('Harga')
plt.legend()
plt.grid(True)
# plt.show() akan menampilkan plot dan menjeda script.
# Gunakan plt.savefig() jika ingin menyimpan ke file tanpa menjeda.
# plt.savefig('plot_perbandingan_harga.png')
plt.show() 

# Plot 2: Residual Plot
print("...Menampilkan Plot 2: Residual Plot.")
residuals_gb = y_test - y_pred_gb
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_gb, y=residuals_gb, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Error = 0')
plt.title('Residual Plot - Gradient Boosting')
plt.xlabel('Nilai Prediksi (Predicted Prices)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.legend()
plt.grid(True)
# plt.savefig('plot_residual.png')
plt.show()

print("\n=== Proses Selesai ===")