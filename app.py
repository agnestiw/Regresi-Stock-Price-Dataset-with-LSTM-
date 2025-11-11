import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import warnings
import datetime

# Menonaktifkan peringatan yang mungkin muncul (opsional)
warnings.filterwarnings('ignore')

# --- 1. Pengaturan Halaman & Judul ---
st.set_page_config(layout="wide", page_title="Prediksi Harga Saham")
st.title("Dashboard Prediksi Harga Saham (Tugas Regresi)")

# --- 2. Fungsi Caching untuk Performa ---

@st.cache_data # Cache data yang dimuat
def load_data(file_path):
    """Memuat data dari file CSV."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"ERROR: File not found at {file_path}")
        st.error("Pastikan file 'Stock_Price_Dataset.csv' berada di folder yang sama dengan script ini.")
        return None

@st.cache_data # Cache data yang diproses
def preprocess_data(df):
    """Melakukan feature engineering pada dataframe."""
    df_processed = df.copy()
    # 1. Ubah 'Date' menjadi angka (ordinal)
    df_processed['Date'] = pd.to_datetime(df_processed['Date'])
    df_processed['Date_ordinal'] = df_processed['Date'].apply(lambda x: x.toordinal())
    
    # 2. Ubah kolom kategori ('Stock', 'Risk') menjadi dummy variables
    df_processed = pd.get_dummies(df_processed, columns=['Stock', 'Risk'], drop_first=True)
    
    # Hapus kolom asli yang tidak lagi diperlukan (Date)
    df_processed = df_processed.drop('Date', axis=1)
    
    # Tentukan Fitur (X) dan Target (y)
    y = df_processed['Price']
    X = df_processed.drop('Price', axis=1)
    return X, y

@st.cache_resource # Cache model yang sudah dilatih
def train_models(X, y):
    """Melatih model AdaBoost dan Gradient Boosting."""
    # Pisahkan Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model 1: AdaBoost
    model_ada = AdaBoostRegressor(random_state=42)
    model_ada.fit(X_train, y_train)
    
    # Model 2: Gradient Boosting
    model_gb = GradientBoostingRegressor(random_state=42)
    model_gb.fit(X_train, y_train)
    
    return X_train, X_test, y_train, y_test, model_ada, model_gb

# --- 3. Fungsi Tampilan Streamlit ---

def display_evaluation(model_name, y_test, y_pred):
    """Menampilkan 5 metrik evaluasi di Streamlit menggunakan st.metric."""
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    st.subheader(f"Hasil Evaluasi: {model_name}")
    
    # Tampilkan metrik utama dalam kolom
    col1, col2, col3 = st.columns(3)
    col1.metric("R-squared (R²)", f"{r2:.3f}")
    col2.metric("RMSE", f"{rmse:.3f}")
    col3.metric("MAPE", f"{mape:.3%}")
    
    # Tampilkan metrik lain di expander
    with st.expander("Lihat metrik lainnya (MAE & MSE)"):
        st.metric("Mean Absolute Error (MAE)", f"{mae:.3f}")
        st.metric("Mean Squared Error (MSE)", f"{mse:.3f}")

# --- 4. Alur Utama Aplikasi ---

# Muat data
df = load_data('Stock_Price_Dataset.csv')

if df is not None:
    # --- 5. Sidebar untuk Input Prediksi ---
    st.sidebar.header("📈 Prediksi Manual")
    
    # Input Tanggal
    default_date = datetime.date(2025, 11, 12)
    input_date = st.sidebar.date_input("Masukkan Tanggal", default_date)
    
    # Input Harga Open
    input_open = st.sidebar.number_input("Masukkan Harga Open", min_value=0.0, value=215.5, step=0.01, format="%.2f")
    
    # Input Ticker Saham (menggunakan selectbox, jauh lebih baik)
    stock_options = df['Stock'].unique()
    # Cari index default untuk 'NVDA' menggunakan numpy
    stock_default_idx_arr = np.where(stock_options == 'NVDA')[0]
    stock_default_idx = int(stock_default_idx_arr[0]) if stock_default_idx_arr.size > 0 else 0
    input_stock = st.sidebar.selectbox("Masukkan Ticker Saham", stock_options, index=stock_default_idx)
    
    # Input Level Risiko
    risk_options = df['Risk'].unique()
    # Cari index default untuk 'high' menggunakan numpy
    risk_default_idx_arr = np.where(risk_options == 'high')[0]
    risk_default_idx = int(risk_default_idx_arr[0]) if risk_default_idx_arr.size > 0 else 0
    input_risk = st.sidebar.selectbox("Masukkan Level Risiko", risk_options, index=risk_default_idx)
    # Tombol untuk menjalankan prediksi
    predict_button = st.sidebar.button("Jalankan Prediksi", type="primary")

    # --- 6. Proses Data dan Latih Model ---
    # (Ini semua berjalan cepat berkat caching)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test, model_ada, model_gb = train_models(X, y)

    # --- 7. Tampilkan Hasil Prediksi Manual (Jika Tombol Ditekan) ---
    if predict_button:
        # Buat dictionary dari input
        data_baru = {
            'Date': pd.to_datetime(input_date),
            'Open': input_open,
            'Stock': input_stock,
            'Risk': input_risk
        }
        
        # Ubah data baru menjadi DataFrame
        df_baru = pd.DataFrame([data_baru])
        
        # Lakukan PROSES FEATURE ENGINEERING yang SAMA PERSIS
        try:
            df_baru['Date_ordinal'] = df_baru['Date'].apply(lambda x: x.toordinal())
            df_baru = pd.get_dummies(df_baru, columns=['Stock', 'Risk'])
            df_baru = df_baru.drop('Date', axis=1)
            
            # PASTIKAN SEMUA KOLOM SAMA DENGAN DATA TRAINING
            df_prediksi = df_baru.reindex(columns=X.columns, fill_value=0)
            
            # Lakukan Prediksi
            hasil_prediksi_gb = model_gb.predict(df_prediksi)
            
            st.sidebar.subheader("--- HASIL PREDIKSI ---")
            result_text = f"${hasil_prediksi_gb[0]:.2f}"
            st.sidebar.success(f"Prediksi harga untuk {input_stock}:")
            st.sidebar.title(result_text)

        except Exception as e:
            st.sidebar.error(f"Error saat prediksi: {e}")

    # --- 8. Tampilan Utama Dashboard ---
    
    # Bagian 1: Tampilan Data
    st.header("1. Eksplorasi Data Awal")
    st.write("Berikut adalah 5 baris pertama dari dataset Anda.")
    st.dataframe(df.head())
    
    # with st.expander("Lihat 5 baris data setelah diproses (Feature Engineering)"):
    #     st.dataframe(X.head())

    # Bagian 2: Evaluasi Model
    st.header("2. Hasil Training & Evaluasi Model")
    st.info("Berikut adalah hasil evaluasi dari 2 model pada 20% data tes.")
    
    # Evaluasi AdaBoost
    y_pred_ada = model_ada.predict(X_test)
    display_evaluation("AdaBoost Regressor", y_test, y_pred_ada)
    
    # Evaluasi Gradient Boosting
    y_pred_gb = model_gb.predict(X_test)
    display_evaluation("Gradient Boosting Regressor", y_test, y_pred_gb)
    
    # Bagian 3: Visualisasi & Analisis Model
    st.header("3. Analisis Model (Gradient Boosting)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Plot 1: Feature Importance
        st.subheader("Feature Importance")
        importances = model_gb.feature_importances_
        feature_imp_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Buat figur Matplotlib baru
        fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(10), ax=ax_imp) # Ambil top 10
        ax_imp.set_title('Top 10 Feature Importance - Gradient Boosting')
        st.pyplot(fig_imp)
        
        with st.expander("Lihat data Feature Importance lengkap"):
            st.dataframe(feature_imp_df)

    with col2:
        # Plot 2: Residual Plot
        st.subheader("Residual Plot")
        residuals_gb = y_test - y_pred_gb
        
        fig_res, ax_res = plt.subplots(figsize=(10, 8))
        sns.scatterplot(x=y_pred_gb, y=residuals_gb, alpha=0.6, ax=ax_res)
        ax_res.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Error = 0')
        ax_res.set_title('Residual Plot - Gradient Boosting')
        ax_res.set_xlabel('Nilai Prediksi (Predicted Prices)')
        ax_res.set_ylabel('Residuals (Actual - Predicted)')
        ax_res.legend()
        ax_res.grid(True)
        st.pyplot(fig_res)

    # Plot 3: Perbandingan Harga (Lebar penuh)
    st.header("4. Visualisasi Perbandingan Harga Aktual vs Prediksi")
    fig_comp, ax_comp = plt.subplots(figsize=(14, 7))
    
    # Urutkan berdasarkan tanggal untuk plot garis yang rapi
    test_data_sorted = X_test.copy()
    test_data_sorted['y_test'] = y_test
    test_data_sorted['y_pred_gb'] = y_pred_gb
    test_data_sorted = test_data_sorted.sort_values(by='Date_ordinal')
    
    ax_comp.scatter(test_data_sorted['Date_ordinal'], test_data_sorted['y_test'], alpha=0.6, color='blue', label='Harga Aktual (Data Tes)', s=10)
    ax_comp.plot(test_data_sorted['Date_ordinal'], test_data_sorted['y_pred_gb'], 
               alpha=0.8, color='red', label='Garis Prediksi (Gradient Boosting)', linewidth=2)
    ax_comp.set_title('Perbandingan Harga Aktual vs Prediksi Model (Gradient Boosting)')
    ax_comp.set_xlabel('Hari (Ordinal)')
    ax_comp.set_ylabel('Harga')
    ax_comp.legend()
    ax_comp.grid(True)
    st.pyplot(fig_comp)