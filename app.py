import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model bundle
bundle = joblib.load("model_bundle.pkl")
model = bundle["model"]
scaler = bundle["scaler"]

st.title("🔋 EV Battery Health Prediction System")
st.write("Masukkan data komponen baterai di bawah ini untuk memprediksi kondisi kesehatan baterai (State of Health / SoH).")

# ------- UI INPUT FORM --------
battery_temp = st.slider("Battery Temperature (°C)", min_value=-10, max_value=100, value=30)
battery_voltage = st.number_input("Battery Voltage (V)", min_value=0.0, value=300.0)
battery_current = st.number_input("Battery Current (A)", min_value=0.0, value=100.0)
power_consumption = st.number_input("Power Consumption (kW)", min_value=0.0, value=10.0)
charge_cycles = st.number_input("Charge Cycles", min_value=0, value=150)

# Create dataframe input
input_df = pd.DataFrame([{
    "Battery_Temperature": battery_temp,
    "Battery_Voltage": battery_voltage,
    "Battery_Current": battery_current,
    "Power_Consumption": power_consumption,
    "Charge_Cycles": charge_cycles
}])

# Scale numeric data
scaled_input = scaler.transform(input_df)

# Prediction button
if st.button("🚗 Predict Battery Health"):
    prediction = model.predict(scaled_input)[0]
    prediction_rounded = round(prediction, 2)

    st.subheader(f"📌 Predicted Battery Health Score: **{prediction_rounded}%**")

    # Status interpretation
    if prediction_rounded >= 80:
        status = "🟢 EXCELLENT — Battery is in very good condition."
    elif prediction_rounded >= 65:
        status = "🟡 FAIR — Battery is still usable but starting to degrade."
    else:
        status = "🔴 POOR — Battery should be maintained or replaced."

    st.write(status)

st.markdown("---")
st.caption("Built by Peter • Machine Learning + Streamlit Deployment")
