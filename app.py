import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="EV Battery Health Prediction",
    page_icon="🔋",
    layout="wide"
)

BATTERY_FEATURES = [
    "Battery_Temperature",
    "Battery_Voltage",
    "Battery_Current",
    "Power_Consumption",
    "Charge_Cycles"
]


# =========================
# FUNGSI: LOAD & PREPROCESS
# =========================
@st.cache_data
def load_raw_data(path: str) -> pd.DataFrame:
    """Load data mentah dari CSV."""
    df = pd.read_csv(path)

    # Drop duplikat
    df = df.drop_duplicates().reset_index(drop=True)

    # Timestamp ke datetime (kalau gagal jadi NaT, tidak apa-apa)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    return df


def iqr_filter(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Filter outlier dengan IQR seperti di ev_predict.py."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]


@st.cache_data
def preprocess_data(path: str) -> pd.DataFrame:
    """
    Preprocessing utama:
    - load CSV
    - drop duplicates
    - parsing Timestamp
    - IQR outlier handling untuk fitur baterai utama
    - feature engineering Health_Score dari Charge_Cycles
    - augment cycle 1–20 dengan Health_Score = 100
    """
    df = load_raw_data(path).copy()

    # Pastikan kolom-kolom fitur baterai ada
    missing_cols = [c for c in BATTERY_FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom berikut tidak ditemukan di dataset: {missing_cols}")

    # OUTLIER HANDLING (IQR per kolom)
    for col in BATTERY_FEATURES:
        before = df.shape[0]
        df = iqr_filter(df, col)
        after = df.shape[0]
        # Bisa dicek via log jika mau:
        # print(f"Outlier removed in {col}: {before - after}")

    df = df.reset_index(drop=True)

    # ===== FEATURE ENGINEERING: HEALTH_SCORE dari Charge_Cycles =====
    tmp = df.copy()
    min_c = tmp["Charge_Cycles"].min()
    max_c = tmp["Charge_Cycles"].max()

    def soh_from_cycle(c):
        c = float(c)

        if c <= 0:
            return 100.0

        if c <= min_c:
            return 100.0

        if c >= max_c:
            return 40.0

        # interpolasi linier antara 95 -> 40
        ratio = (c - min_c) / (max_c - min_c)
        return 95.0 - ratio * 55.0

    tmp["Health_Score"] = tmp["Charge_Cycles"].apply(soh_from_cycle)
    tmp["Health_Score"] = tmp["Health_Score"].clip(0, 100)

    # ===== AUGMENTASI: cycle 1–20 dengan Health_Score = 100 =====
    feature_median = tmp[BATTERY_FEATURES].median()

    synthetic_rows = []
    for c in range(1, 21):
        row = feature_median.copy()
        row["Charge_Cycles"] = c
        synthetic_rows.append(row)

    synthetic_df = pd.DataFrame(synthetic_rows)
    synthetic_df["Health_Score"] = 100.0

    df_aug = pd.concat([tmp, synthetic_df], ignore_index=True)

    return df_aug


# =========================
# FUNGSI: TRAIN MODEL
# =========================
@st.cache_resource
def train_model(df: pd.DataFrame):
    """
    Melatih RandomForestRegressor dengan StandardScaler
    mengacu pada script ev_predict.py.
    Mengembalikan: model, scaler, metrics.
    """
    X = df[BATTERY_FEATURES]
    y = df["Health_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)

    # Evaluasi (opsional, untuk ditampilkan di UI)
    y_pred = rf.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    return rf, scaler, metrics


def classify_condition(soh: float) -> str:
    """Mapping kondisi baterai berdasarkan Health_Score (SoH)."""
    if soh >= 80:
        return "SEHAT"
    elif soh >= 70:
        return "CUKUP SEHAT"
    else:
        return "BURUK"


# =========================
# UI STREAMLIT
# =========================
def main():
    st.title("🔋 EV Battery Health Prediction (SoH)")
    st.markdown(
        """
        Aplikasi ini menggunakan model **Random Forest Regression** 
        untuk memprediksi _State of Health (SoH)_ baterai EV berdasarkan:

        - Battery Temperature  
        - Battery Voltage  
        - Battery Current  
        - Power Consumption  
        - Charge Cycles  

        Preprocessing dan feature engineering mengikuti script analisis di PKL/Colab kamu.
        """
    )

    # ---- Sidebar: Info Dataset ----
    st.sidebar.header("Pengaturan")
    data_path = st.sidebar.text_input(
        "Path dataset CSV:",
        value="EV_Predictive_Maintenance_Dataset_15min.csv",
        help="Pastikan nama file sama dengan di folder proyek."
    )

    try:
        df_processed = preprocess_data(data_path)
    except Exception as e:
        st.error(f"Gagal memproses data: {e}")
        st.stop()

    # Train model
    rf_model, scaler, metrics = train_model(df_processed)

    # ---- Tampilkan ringkasan dataset ----
    st.subheader("Ringkasan Dataset Setelah Preprocessing")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Ukuran data (setelah outlier handling & augmentasi):")
        st.write(df_processed.shape)
        st.write("Beberapa baris pertama:")
        st.dataframe(df_processed[BATTERY_FEATURES + ["Health_Score"]].head())

    with col2:
        st.write("Statistik Health_Score:")
        st.write(df_processed["Health_Score"].describe())

        st.write("Performa Model (Random Forest):")
        st.json(metrics)

    st.markdown("---")

    # ---- Form Input Prediksi ----
    st.subheader("Masukkan Data Baterai untuk Prediksi SoH")

    median_values = df_processed[BATTERY_FEATURES].median()

    c1, c2, c3 = st.columns(3)

    with c1:
        temp = st.number_input(
            "Battery Temperature (°C)",
            value=float(median_values["Battery_Temperature"])
        )
        volt = st.number_input(
            "Battery Voltage (V)",
            value=float(median_values["Battery_Voltage"])
        )

    with c2:
        curr = st.number_input(
            "Battery Current (A)",
            value=float(median_values["Battery_Current"])
        )
        power = st.number_input(
            "Power Consumption (kW atau sesuai unit dataset)",
            value=float(median_values["Power_Consumption"])
        )

    with c3:
        cycles = st.number_input(
            "Charge Cycles",
            min_value=0.0,
            value=float(median_values["Charge_Cycles"])
        )

    if st.button("🔮 Prediksi Kesehatan Baterai"):
        user_df = pd.DataFrame([{
            "Battery_Temperature": temp,
            "Battery_Voltage": volt,
            "Battery_Current": curr,
            "Power_Consumption": power,
            "Charge_Cycles": cycles
        }])

        user_scaled = scaler.transform(user_df)
        pred_soh = rf_model.predict(user_scaled)[0]
        kondisi = classify_condition(pred_soh)

        st.success(f"Perkiraan State of Health (SoH): **{pred_soh:.2f}%**")
        if kondisi == "SEHAT":
            st.markdown(f"✅ Kondisi baterai: **{kondisi}**")
        elif kondisi == "CUKUP SEHAT":
            st.markdown(f"🟡 Kondisi baterai: **{kondisi}**")
        else:
            st.markdown(f"🔴 Kondisi baterai: **{kondisi}**")

        st.markdown(
            """
            **Interpretasi singkat:**
            - ≥ 80% : baterai masih sangat layak pakai  
            - 70–79% : mulai menurun, perlu pemantauan  
            - < 70% : kondisi cukup buruk, pertimbangkan perawatan/penggantian  
            """
        )


if __name__ == "__main__":
    main()
