from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

# ==========================
# Load model files
# ==========================

model = joblib.load("itad_rf_model.pkl")
encoders = joblib.load("itad_encoders.pkl")
features = joblib.load("itad_features.pkl")

# ==========================
# Flask app
# ==========================

app = Flask(__name__)
CORS(app)

# ==========================
# Helper functions
# ==========================

def categorize(name):
    name = str(name).lower()

    if any(k in name for k in ['laptop', 'computer', 'pc', 'desktop']):
        return 'computer'

    if any(k in name for k in ['printer', 'fax', 'scanner']):
        return 'printer'

    if any(k in name for k in ['projector', 'lcd', 'tv', 'screen', 'monitor']):
        return 'display'

    if any(k in name for k in ['server', 'switch', 'router']):
        return 'network'

    return 'other'


def safe_encode(column, value):
    le = encoders[column]
    value = str(value)

    if value in le.classes_:
        return le.transform([value])[0]

    return 0


life_map = {
    "computer": 5,
    "printer": 6,
    "display": 7,
    "network": 6,
    "other": 6
}

warranty_map = {
    "computer": 3,
    "printer": 2,
    "display": 3,
    "network": 3,
    "other": 2
}

# ==========================
# Routes
# ==========================

@app.route("/")
def home():
    return "ITAD AI API is running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        item_name = data.get("item_name", "Unknown")
        price = float(data.get("price", 0))
        age = float(data.get("item_age_years", 0))

        category = categorize(item_name)

        default_life = life_map.get(category, 6)
        warranty = warranty_map.get(category, 2)

        out_of_warranty = int(age > warranty)
        exceeded_life = int(age > default_life)

        current_value = price * (1 - (age / default_life) * 0.6)
        current_value = max(current_value, 1)

        repair_cost = price * 0.25
        repair_ratio = repair_cost / current_value

        item_name_encoded = safe_encode("item_name", item_name)
        category_encoded = safe_encode("item_category", category)

        sample = pd.DataFrame([{
            "item_name": item_name_encoded,
            "item_category": category_encoded,
            "price": price,
            "item_age_years": age,
            "default_life_years": default_life,
            "warranty_years": warranty,
            "out_of_warranty": out_of_warranty,
            "exceeded_life": exceeded_life,
            "current_value": current_value,
            "estimated_repair_cost": repair_cost,
            "repair_cost_ratio": repair_ratio
        }])

        sample = sample[features]

        prediction_code = model.predict(sample)[0]
        prediction_label = encoders["itad_decision"].classes_[prediction_code]

        return jsonify({
            "prediction": prediction_label,
            "item_category": category,
            "current_value": round(current_value, 2),
            "repair_cost_ratio": round(repair_ratio, 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
