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
# Device categorization
# ==========================

def categorize(name):
    name = str(name).lower()

    if any(k in name for k in ["laptop", "notebook", "macbook"]):
        return "computer"

    if any(k in name for k in ["desktop", "pc", "optiplex", "workstation"]):
        return "desktop"

    if any(k in name for k in ["printer", "laserjet", "inkjet", "fax"]):
        return "printer"

    if any(k in name for k in ["scanner", "scanjet"]):
        return "scanner"

    if any(k in name for k in ["monitor", "lcd", "led", "screen", "display", "projector"]):
        return "display"

    if any(k in name for k in ["router", "switch", "cisco", "netgear", "access point", "network"]):
        return "network"

    if any(k in name for k in ["server", "rack", "storage", "nas"]):
        return "server"

    if any(k in name for k in ["phone", "mobile", "tablet", "ipad", "iphone"]):
        return "phone"

    if any(k in name for k in ["camera", "speaker", "headset", "microphone", "audio"]):
        return "av_equipment"

    if any(k in name for k in ["fan", "ac", "air", "cooler", "hvac"]):
        return "hvac"

    return "other"


# ==========================
# Safe encoder
# ==========================

def safe_encode(column, value):
    le = encoders[column]
    value = str(value)

    if value in le.classes_:
        return le.transform([value])[0]

    return 0


# ==========================
# Useful life and warranty
# ==========================

life_map = {
    "computer": 5,
    "desktop": 6,
    "printer": 6,
    "scanner": 6,
    "display": 7,
    "network": 6,
    "server": 7,
    "phone": 4,
    "av_equipment": 5,
    "hvac": 10,
    "other": 6
}

warranty_map = {
    "computer": 3,
    "desktop": 3,
    "printer": 2,
    "scanner": 2,
    "display": 3,
    "network": 3,
    "server": 4,
    "phone": 2,
    "av_equipment": 2,
    "hvac": 5,
    "other": 2
}


# ==========================
# Component analysis engine
# ==========================

def component_analysis(category, condition, age, repair_ratio):
    condition = str(condition).lower()

    components = {
        "computer": {
            "valuable_components": ["RAM", "SSD/HDD", "Screen", "Battery", "Motherboard"],
            "reusable_parts": ["RAM", "SSD/HDD", "Charger", "Keyboard"],
            "recyclable_materials": ["Aluminum", "Copper", "Plastic", "Circuit Boards"],
            "hazardous_parts": ["Lithium Battery"]
        },
        "desktop": {
            "valuable_components": ["CPU", "RAM", "SSD/HDD", "Power Supply", "Motherboard", "GPU"],
            "reusable_parts": ["RAM", "SSD/HDD", "Power Supply", "Cooling Fan"],
            "recyclable_materials": ["Steel", "Aluminum", "Copper", "Plastic", "Circuit Boards"],
            "hazardous_parts": ["CMOS Battery"]
        },
        "printer": {
            "valuable_components": ["Toner Cartridge", "Printer Motor", "Power Supply", "Scanner Unit"],
            "reusable_parts": ["Toner Cartridge", "Paper Tray", "Power Supply"],
            "recyclable_materials": ["Plastic", "Steel", "Copper Wiring", "Circuit Boards"],
            "hazardous_parts": ["Toner Residue", "Ink Waste"]
        },
        "scanner": {
            "valuable_components": ["Scanner Sensor", "Glass Panel", "Power Supply", "Motor"],
            "reusable_parts": ["Glass Panel", "Power Supply", "USB Cable"],
            "recyclable_materials": ["Glass", "Plastic", "Copper", "Circuit Boards"],
            "hazardous_parts": ["Lamp Unit"]
        },
        "display": {
            "valuable_components": ["LCD/LED Panel", "Power Board", "Backlight", "Stand"],
            "reusable_parts": ["Stand", "Cables", "Power Board"],
            "recyclable_materials": ["Glass", "Plastic", "Copper", "Aluminum"],
            "hazardous_parts": ["Backlight Components"]
        },
        "network": {
            "valuable_components": ["Network Board", "Power Adapter", "Ports", "Antennas"],
            "reusable_parts": ["Power Adapter", "Antennas", "Cables"],
            "recyclable_materials": ["Plastic", "Copper", "Circuit Boards", "Metal Ports"],
            "hazardous_parts": ["Small Capacitors"]
        },
        "server": {
            "valuable_components": ["CPU", "RAM", "Storage Drives", "Power Supply", "RAID Controller", "Network Cards"],
            "reusable_parts": ["RAM", "Storage Drives", "Power Supply", "Cooling Fans", "Network Cards"],
            "recyclable_materials": ["Steel", "Aluminum", "Copper", "Circuit Boards"],
            "hazardous_parts": ["CMOS Battery", "Backup Battery"]
        },
        "phone": {
            "valuable_components": ["Screen", "Battery", "Camera Module", "Storage Chip", "Logic Board"],
            "reusable_parts": ["Screen", "Camera Module", "Charger"],
            "recyclable_materials": ["Glass", "Aluminum", "Copper", "Rare Metals"],
            "hazardous_parts": ["Lithium Battery"]
        },
        "av_equipment": {
            "valuable_components": ["Speaker Unit", "Camera Lens", "Microphone", "Control Board"],
            "reusable_parts": ["Cables", "Speaker Unit", "Microphone"],
            "recyclable_materials": ["Plastic", "Copper", "Magnets", "Circuit Boards"],
            "hazardous_parts": ["Small Capacitors"]
        },
        "hvac": {
            "valuable_components": ["Compressor", "Fan Motor", "Copper Coil", "Control Board"],
            "reusable_parts": ["Fan Motor", "Control Board", "Copper Coil"],
            "recyclable_materials": ["Copper", "Aluminum", "Steel", "Plastic"],
            "hazardous_parts": ["Refrigerant Gas", "Capacitors"]
        },
        "other": {
            "valuable_components": ["Circuit Board", "Power Supply", "Cables"],
            "reusable_parts": ["Cables", "Power Adapter"],
            "recyclable_materials": ["Plastic", "Copper", "Metal", "Circuit Boards"],
            "hazardous_parts": ["Capacitors"]
        }
    }

    result = components.get(category, components["other"]).copy()

    if condition in ["poor", "damage", "damaged"] or repair_ratio > 1.0:
        result["reuse_recommendation"] = "Limited reuse. Prefer component harvesting and recycling."
    elif age <= 3:
        result["reuse_recommendation"] = "High reuse potential. Device/components may be reused or resold."
    elif age <= 6:
        result["reuse_recommendation"] = "Moderate reuse potential. Inspect components before reuse."
    else:
        result["reuse_recommendation"] = "Low device reuse potential. Valuable components may still be recovered."

    return result


# ==========================
# Explanation generator
# ==========================

def generate_explanation(prediction, category, age, default_life, warranty, repair_ratio):
    reasons = []

    if age > default_life:
        reasons.append("device age exceeds the expected useful life")

    if age > warranty:
        reasons.append("device is outside the estimated warranty period")

    if repair_ratio > 1:
        reasons.append("repair cost ratio is high compared to current value")

    if not reasons:
        reasons.append("device is still within acceptable lifecycle limits")

    return (
        f"The device was categorized as {category}. "
        f"The model predicted '{prediction}' because " + ", ".join(reasons) + "."
    )


# ==========================
# Routes
# ==========================

@app.route("/")
def home():
    return "ITAD AI API with component analysis is running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json or {}

        item_name = data.get("item_name", "Unknown")
        price = float(data.get("price", 0))
        age = float(data.get("item_age_years", 0))
        condition = data.get("condition", "Unknown")

        category = categorize(item_name)

        default_life = life_map.get(category, 6)
        warranty = warranty_map.get(category, 2)

        out_of_warranty = int(age > warranty)
        exceeded_life = int(age > default_life)

        current_value = price * (1 - (age / default_life) * 0.6)
        current_value = max(current_value, 1)

        repair_cost = price * 0.25
        repair_ratio = repair_cost / current_value

        # Important:
        # Model was trained mostly with categories such as computer/printer/display/network/other.
        # If category was not seen before, safe_encode prevents crashing.
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

        components = component_analysis(
            category=category,
            condition=condition,
            age=age,
            repair_ratio=repair_ratio
        )

        explanation = generate_explanation(
            prediction=prediction_label,
            category=category,
            age=age,
            default_life=default_life,
            warranty=warranty,
            repair_ratio=repair_ratio
        )

        return jsonify({
            "prediction": prediction_label,
            "item_category": category,
            "current_value": round(current_value, 2),
            "estimated_repair_cost": round(repair_cost, 2),
            "repair_cost_ratio": round(repair_ratio, 2),
            "default_life_years": default_life,
            "warranty_years": warranty,
            "out_of_warranty": out_of_warranty,
            "exceeded_life": exceeded_life,
            "components": components,
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
