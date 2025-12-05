import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# ===========================
# 🔧 AUTO LOAD / TRAIN MODEL
# ===========================
@st.cache_resource
def load_model():
    model_file = "model.pkl"
    scaler_file = "scaler.pkl"

    # If model already exists → load it
    if os.path.exists(model_file) and os.path.exists(scaler_file):
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        return model, scaler

    # Otherwise train from dataset
    dataset = None
    for f in os.listdir():
        if f.endswith(".csv"):
            dataset = f
            break

    if dataset is None:
        st.error("❌ Dataset CSV tidak ditemukan dalam folder!")
        return None, None

    st.warning(f"📂 Dataset ditemukan: **{dataset}** — Training model... ⏳")

    df = pd.read_csv(dataset)

    # Select features & target (sesuaikan dengan dataset kamu!)
    X = df.drop("Depression", axis=1)
    y = df["Depression"]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    st.success("🎉 Model berhasil dilatih dan disimpan!")

    return model, scaler


model, scaler = load_model()



# ===========================
# 🎨 STREAMLIT UI
# ===========================
st.title("🧠 Mental Health Depression Prediction App")
st.write("Masukkan data pengguna untuk memprediksi kemungkinan depresi.")


# If model not loaded
if model is None:
    st.stop()


# ===========================
# 📝 User Input Form
# ===========================
st.subheader("📌 Isi Data Berikut:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=20)
    academic_pressure = st.slider("Academic Pressure", 1, 10, 5)
    study_hours = st.slider("Study / Work Hours", 1, 12, 6)

with col2:
    sleep_duration = st.slider("Sleep Duration (hours)", 1, 10, 6)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0)
    financial_stress = st.slider("Financial Stress", 1, 10, 5)

input_data = pd.DataFrame([[age, academic_pressure, study_hours, sleep_duration, cgpa, financial_stress]],
                          columns=["Age", "AcademicPressure", "StudyHours", "SleepDuration", "CGPA", "FinancialStress"])


# ===========================
# 🔍 Predict Button
# ===========================
if st.button("🔮 Predict"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠ Hasil: **Berpotensi Depresi**. Sebaiknya konsultasi lebih lanjut. 💛")
    else:
        st.success("😄 Hasil: **Tidak Depresi**. Tetap jaga kesehatan mental!")


st.info("💡 Model menggunakan Random Forest dan preprocessing otomatis dari dataset CSV.")
