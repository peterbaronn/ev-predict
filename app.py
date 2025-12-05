import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# KONFIGURASI PAGE
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
# LOAD MODEL & SCALER
# =========================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("rf_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Gagal load model/scaler: {e}")
        st.stop()


model, scaler = load_model()


# =========================
# FUNGSI KLASIFIKASI KONDISI
# =========================
def classify_condition(soh: float) -> str:
    """Mapping kondisi baterai berdasarkan SoH."""
    if soh >= 80:
        return "SEHAT"
    elif soh >= 70:
        return "CUKUP SEHAT"
    else:
        return "BURUK"


# =========================
# USER INTERFACE
# =========================
def main():

    st.title("🔋 EV Battery Health Prediction (Model .PKL)")
    st.write("""
        Aplikasi ini menggunakan model produksi **Random Forest (.pkl)**  
        untuk memprediksi **State of Health (SoH)** baterai EV secara cepat.
    """)

    st.markdown("---")

    # =========================
    # MODE INPUT: MANUAL & FILE
    # =========================
    mode = st.radio("Pilih metode input:", ["🔧 Input Manual", "📁 Upload CSV"])

    # ---- MODE 1: INPUT MANUAL ----
    if mode == "🔧 Input Manual":

        st.subheader("Masukkan Data Untuk Prediksi")

        col1, col2, col3 = st.columns(3)

        with col1:
            temp = st.number_input("Battery Temperature (°C)", value=25.0)
            voltage = st.number_input("Battery Voltage (V)", value=350.0)

        with col2:
            current = st.number_input("Battery Current (A)", value=50.0)
            power = st.number_input("Power Consumption (kW atau unit dataset)", value=10.0)

        with col3:
            cycles = st.number_input("Charge Cycles", min_value=0.0, value=100.0)

        st.markdown("---")

        if st.button("🔮 Prediksi SoH (Manual)"):
            input_df = pd.DataFrame([{
                "Battery_Temperature": temp,
                "Battery_Voltage": voltage,
                "Battery_Current": current,
                "Power_Consumption": power,
                "Charge_Cycles": cycles
            }])

            scaled = scaler.transform(input_df)
            soh = model.predict(scaled)[0]
            kondisi = classify_condition(soh)

            st.success(f"Hasil Prediksi SoH: **{soh:.2f}%**")

            if kondisi == "SEHAT":
                st.info("⚡ Status: **SEHAT — Baterai dalam kondisi baik.**")
            elif kondisi == "CUKUP SEHAT":
                st.warning("🟡 Status: **CUKUP SEHAT — Perlu pemantauan.**")
            else:
                st.error("🔴 Status: **BURUK — Perlu perawatan atau penggantian.**")


    # ---- MODE 2: UPLOAD CSV ----
    elif mode == "📁 Upload CSV":

        st.subheader("Upload File CSV Untuk Prediksi Banyak Data")
        file = st.file_uploader("Unggah file CSV", type=["csv"])

        if file:
            df = pd.read_csv(file)

            # Validasi format
            missing_cols = [c for c in BATTERY_FEATURES if c not in df.columns]
            if missing_cols:
                st.error(f"Kolom berikut tidak ditemukan: {missing_cols}")
                return

            # Prediksi
            scaled = scaler.transform(df[BATTERY_FEATURES])
            df["Predicted_SoH"] = model.predict(scaled)
            df["Condition"] = df["Predicted_SoH"].apply(classify_condition)

            st.success("Prediksi berhasil dilakukan!")
            st.dataframe(df)

            # Export file hasil
            csv_result = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Hasil Prediksi",
                data=csv_result,
                file_name="EV_Battery_Prediction_Result.csv",
                mime="text/csv",
            )


    # -------------------------------
    # Informasi Batas Interpretasi
    # -------------------------------
    st.markdown("""
    ---
    ### 📌 Panduan Interpretasi SoH
    | SoH (%) | Kondisi |
    |---------|---------|
    | ≥ 80 | Sangat sehat, layak operasi |
    | 70–79 | Penurunan mulai terasa |
    | < 70 | Risiko performa buruk, periksa lebih lanjut |
    """)


if __name__ == "__main__":
    main()
