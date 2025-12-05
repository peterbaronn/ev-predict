import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# ==========================
# Load or Train Model
# ==========================
@st.cache_resource
def load_model():

    model_file = "ev_model.pkl"
    scaler_file = "ev_scaler.pkl"

    # If model already exists → load it
    if os.path.exists(model_file) and os.path.exists(scaler_file):
        return joblib.load(model_file), joblib.load(scaler_file)

    # Otherwise → train model from dataset
    dataset = "EV_Predictive_Maintenance_Dataset_15min.csv"

    if not os.path.exists(dataset):
        raise FileNotFoundError(f"Dataset '{dataset}' tidak ditemukan!")

    df = pd.read_csv(dataset)

    # Clean duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Define features & target
    features = [
        "Battery_Temperature",
        "Battery_Voltage",
        "Battery_Current",
        "Power_Consumption",
        "Charge_Cycles"
    ]

    # Feature Engineering: Custom SoH logic
    min_cycle = df["Charge_Cycles"].min()
    max_cycle = df["Charge_Cycles"].max()

    def soh_formula(cycle):
        cycle = float(cycle)
        if cycle <= min_cycle: return 100
        if cycle >= max_cycle: return 40
        ratio = (cycle - min_cycle) / (max_cycle - min_cycle)
        return round(95 - ratio * 55, 2)

    df["Health_Score"] = df["Charge_Cycles"].apply(soh_formula)

    # Train Test Split
    X = df[features]
    y = df["Health_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Model
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save to PKL
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    return model, scaler


# ==========================
# STREAMLIT UI
# ==========================
st.set_page_config(page_title="EV Battery Health Predictor", page_icon="🔋")

st.title("🔋 EV Battery Health Prediction App")
st.write("Masukkan nilai berikut untuk memprediksi kondisi kesehatan baterai (SoH).")

model, scaler = load_model()


# ========= User Input =========
temp = st.number_input("Battery Temperature (°C)", 0.0, 120.0, 35.0)
volt = st.number_input("Battery Voltage (V)", 0.0, 1000.0, 350.0)
current = st.number_input("Battery Current (A)", -300.0, 300.0, 50.0)
power = st.number_input("Power Consumption (kW)", 0.0, 1000.0, 60.0)
cycles = st.number_input("Charge Cycles", 0.0, 5000.0, 500.0)

features = pd.DataFrame([{
    "Battery_Temperature": temp,
    "Battery_Voltage": volt,
    "Battery_Current": current,
    "Power_Consumption": power,
    "Charge_Cycles": cycles
}])


# ========= Predict =========
if st.button("🔮 Predict Battery Health"):
    try:
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]

        if prediction > 80:
            status = "🟢 Excellent - Battery Healthy"
        elif prediction > 70:
            status = "🟡 Moderate - Monitor usage"
        else:
            status = "🔴 Poor - Needs Maintenance"

        st.success(f"🧪 **Predicted SoH: {prediction:.2f}%**")
        st.write(f"Status: {status}")

    except Exception as e:
        st.error(f"Error: {e}")


st.write("---")
st.caption("Made by Peter | Machine Learning • Streamlit • Random Forest")
