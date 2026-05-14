from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib


# ==========================
# Load model files
# ==========================

model = joblib.load(
    "itad_rf_model.pkl"
)

encoders = joblib.load(
    "itad_encoders.pkl"
)

features = joblib.load(
    "itad_features.pkl"
)


# ==========================
# Create Flask app
# ==========================

app = Flask(__name__)
CORS(app)

# ==========================
# Categorize device
# ==========================

def categorize(name):

    name = str(name).lower()


    if any(k in name for k in
           ['laptop','computer','pc','desktop']):
        return 'computer'


    if any(k in name for k in
           ['printer','fax','scanner']):
        return 'printer'


    if any(k in name for k in
           ['projector','lcd','tv','screen']):
        return 'display'


    if any(k in name for k in
           ['server','switch','router']):
        return 'network'


    return 'other'


# ==========================
# Useful life
# ==========================

life_map = {
    "computer": 5,
    "printer": 6,
    "display": 7,
    "network": 6,
    "other": 6
}


# ==========================
# Warranty
# ==========================

warranty_map = {
    "computer": 3,
    "printer": 2,
    "display": 3,
    "network": 3,
    "other": 2
}


# ==========================
# Health check
# ==========================

@app.route("/")

def home():

    return "ITAD AI API is running"


# ==========================
# Prediction
# ==========================

@app.route(
    "/predict",
    methods=["POST"]
)

def predict():

    data = request.json


    item_name = data["item_name"]

    price = float(
        data["price"]
    )

    age = float(
        data["item_age_years"]
    )


    category = categorize(
        item_name
    )


    default_life = life_map[
        category
    ]


    warranty = warranty_map[
        category
    ]


    out_of_warranty = int(
        age > warranty
    )


    exceeded_life = int(
        age > default_life
    )


    current_value = price * (
        1 - (
            age /
            default_life
        ) * 0.6
    )


    if current_value < 1:
        current_value = 1


    repair_cost = (
        price * 0.25
    )


    repair_ratio = (
        repair_cost /
        current_value
    )


    # Encode text columns

    item_name_encoded = encoders[
        "item_name"
    ].transform(
        [item_name]
    )[0]


    category_encoded = encoders[
        "item_category"
    ].transform(
        [category]
    )[0]


    sample = pd.DataFrame([{

        "item_name":
        item_name_encoded,

        "item_category":
        category_encoded,

        "price":
        price,

        "item_age_years":
        age,

        "default_life_years":
        default_life,

        "warranty_years":
        warranty,

        "out_of_warranty":
        out_of_warranty,

        "exceeded_life":
        exceeded_life,

        "current_value":
        current_value,

        "estimated_repair_cost":
        repair_cost,

        "repair_cost_ratio":
        repair_ratio

    }])


    prediction_code = model.predict(
        sample
    )[0]


    prediction_label = encoders[
        "itad_decision"
    ].classes_[
        prediction_code
    ]


    return jsonify({

        "prediction":
        prediction_label

    })


# ==========================
# Run locally
# ==========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8000
    )
