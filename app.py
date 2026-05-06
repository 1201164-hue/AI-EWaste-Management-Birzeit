from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("ewaste_rf_model.pkl")
encoders = joblib.load("ewaste_encoders.pkl")
target_encoder = joblib.load("ewaste_target_encoder.pkl")

FEATURES = [
    "اسم الصنف",
    "سعر الصنف بالدينار",
    "المبنى",
    "الغرفة",
    "الدائرة"
]

@app.route("/")
def home():
    return "E-Waste AI Model API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    row = pd.DataFrame([{
        "اسم الصنف": data.get("item_name", "Unknown"),
        "سعر الصنف بالدينار": data.get("price", 0),
        "المبنى": data.get("building", "Unknown"),
        "الغرفة": data.get("room", "Unknown"),
        "الدائرة": data.get("department", "Unknown")
    }])

    for col in FEATURES:
        row[col] = row[col].astype(str)

        if col in encoders:
            le = encoders[col]

            if row[col].iloc[0] in le.classes_:
                row[col] = le.transform(row[col])
            else:
                row[col] = 0

    prediction = model.predict(row[FEATURES])[0]
    result = target_encoder.inverse_transform([prediction])[0]

    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)