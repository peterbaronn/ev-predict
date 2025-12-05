import streamlit as st
import pandas as pd
import numpy as np

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
# SOH FORMULA (TIDAK PAKAI ML)
# =========================
def calculate_soh(cycle, min_cycle=0, max_cycle=1200):
    """Rumus prediksi SoH tanpa ML."""
    cycle = float(cycle)

    if cycle <= min_cycle:
        return 100.0

    if cycle >= max_cycle:
        return 40.0

    ratio = (cycle - min_cycle) / (max_cycle - min_cycle)
    soh = 95 - (ratio * 55)
    return max(40, min(100, soh))


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

    st.title("🔋 EV Battery Health Prediction")
    st.write("""
        Prediksi **State of Health (SoH)** baterai EV menggunakan
        pendekatan matematis berbasis degradasi siklus baterai,
        **tanpa machine learning**.
    """)

    st.markdown("---")

    mode = st.radio("Pilih metode input:", ["🔧 Input Manual", "📁 Upload CSV"])


    # ---- INPUT MANUAL ----
    if mode == "🔧 Input Manual":

        col1, col2, col3 = st.columns(3)

        with col1:
            temp = st.number_input("Battery Temperature (°C)", value=25.0)
            voltage = st.number_input("Battery Voltage (V)", value=350.0)

        with col2:
            current = st.number_input("Battery Current (A)", value=50.0)
            power = st.number_input("Power Consumption", value=10.0)

        with col3:
            cycles = st.number_input("Charge Cycles", min_value=0.0, value=100.0)

        st.markdown("---")

        if st.button("🔮 Prediksi SoH"):
            soh = calculate_soh(cycles)
            status = classify_condition(soh)

            st.success(f"Hasil Prediksi SoH: **{soh:.2f}%**")

            if status == "SEHAT":
                st.info("⚡ Status: **SEHAT — Baterai dalam kondisi optimal.**")
            elif status == "CUKUP SEHAT":
                st.warning("🟡 Status: **CUKUP SEHAT — Waspadai degradasi.**")
            else:
                st.error("🔴 Status: **BURUK — Perlu pemeriksaan / penggantian.**")


    # ---- CSV UPLOAD ----
    elif mode == "📁 Upload CSV":

        st.subheader("Upload File CSV")
        file = st.file_uploader("Unggah file CSV", type=["csv"])

        if file:
            df = pd.read_csv(file)

            if "Charge_Cycles" not in df.columns:
                st.error("Kolom wajib 'Charge_Cycles' tidak ditemukan.")
                return

            df["Predicted_SoH"] = df["Charge_Cycles"].apply(calculate_soh)
            df["Condition"] = df["Predicted_SoH"].apply(classify_condition)

            st.success("Prediksi berhasil!")
            st.dataframe(df)

            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Hasil Prediksi",
                data=csv_out,
                file_name="EV_Battery_Health_Prediction.csv",
                mime="text/csv"
            )


    # FOOTER INFO
    st.markdown("""
    ---
    ### 📌 Panduan Interpretasi SoH
    | SoH (%) | Kondisi |
    |---------|---------|
    | ≥ 80 | 🟢 Sangat Sehat |
    | 70–79 | 🟡 Mulai Menurun |
    | < 70 | 🔴 Perlu Pemeriksaan |
    """)


if __name__ == "__main__":
    main()
