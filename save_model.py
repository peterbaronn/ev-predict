import joblib

# load model dan scaler dari notebook kamu (ubah ke nama file kamu kalau beda)
model = joblib.load("ev_rf_model.pkl")  # kalau nama beda, ubah
scaler = joblib.load("ev_scaler.pkl")

bundle = {
    "model": model,
    "scaler": scaler
}

joblib.dump(bundle, "model_bundle.pkl")
print("✔ model_bundle.pkl berhasil dibuat!")
