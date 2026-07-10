from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
from openai import OpenAI

# ==========================
# Load model files
# ==========================

model = joblib.load("itad_rf_model.pkl")
encoders = joblib.load("itad_encoders.pkl")
features = joblib.load("itad_features.pkl")

DATASET_PATH = os.environ.get("EWASTE_DATASET_PATH", "ewaste_itad_final.csv")

# ==========================
# Flask app
# ==========================

app = Flask(__name__)
CORS(app)

# ==========================
# Dataset helpers
# ==========================

def load_dataset():
    """Load the final dataset if it exists beside app.py."""
    if not os.path.exists(DATASET_PATH):
        return None
    df = pd.read_csv(DATASET_PATH)
    if "serial_number" in df.columns:
        df["serial_lookup"] = (
            df["serial_number"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.strip()
        )
    return df


def clean_serial(value):
    return str(value).replace(".0", "").strip()

# ==========================
# Device categorization
# ==========================

def categorize(name):
    name = str(name).lower()

    if any(k in name for k in ["laptop", "notebook", "macbook", "latitude", "thinkpad", "hp compaq"]):
        return "computer"
    if any(k in name for k in ["desktop", "pc", "optiplex", "workstation"]):
        return "desktop"
    if any(k in name for k in ["printer", "laserjet", "inkjet", "fax"]):
        return "printer"
    if any(k in name for k in ["scanner", "scanjet"]):
        return "scanner"
    if any(k in name for k in ["monitor", "lcd", "led", "screen", "display", "projector", "benq"]):
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
    "other": 6,
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
    "other": 2,
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
            "hazardous_parts": ["Lithium Battery"],
        },
        "desktop": {
            "valuable_components": ["CPU", "RAM", "SSD/HDD", "Power Supply", "Motherboard", "GPU"],
            "reusable_parts": ["RAM", "SSD/HDD", "Power Supply", "Cooling Fan"],
            "recyclable_materials": ["Steel", "Aluminum", "Copper", "Plastic", "Circuit Boards"],
            "hazardous_parts": ["CMOS Battery"],
        },
        "printer": {
            "valuable_components": ["Toner Cartridge", "Printer Motor", "Power Supply", "Scanner Unit"],
            "reusable_parts": ["Toner Cartridge", "Paper Tray", "Power Supply"],
            "recyclable_materials": ["Plastic", "Steel", "Copper Wiring", "Circuit Boards"],
            "hazardous_parts": ["Toner Residue", "Ink Waste"],
        },
        "scanner": {
            "valuable_components": ["Scanner Sensor", "Glass Panel", "Power Supply", "Motor"],
            "reusable_parts": ["Glass Panel", "Power Supply", "USB Cable"],
            "recyclable_materials": ["Glass", "Plastic", "Copper", "Circuit Boards"],
            "hazardous_parts": ["Lamp Unit"],
        },
        "display": {
            "valuable_components": ["LCD/LED Panel", "Power Board", "Backlight", "Stand"],
            "reusable_parts": ["Stand", "Cables", "Power Board"],
            "recyclable_materials": ["Glass", "Plastic", "Copper", "Aluminum"],
            "hazardous_parts": ["Backlight Components"],
        },
        "network": {
            "valuable_components": ["Network Board", "Power Adapter", "Ports", "Antennas"],
            "reusable_parts": ["Power Adapter", "Antennas", "Cables"],
            "recyclable_materials": ["Plastic", "Copper", "Circuit Boards", "Metal Ports"],
            "hazardous_parts": ["Small Capacitors"],
        },
        "server": {
            "valuable_components": ["CPU", "RAM", "Storage Drives", "Power Supply", "RAID Controller", "Network Cards"],
            "reusable_parts": ["RAM", "Storage Drives", "Power Supply", "Cooling Fans", "Network Cards"],
            "recyclable_materials": ["Steel", "Aluminum", "Copper", "Circuit Boards"],
            "hazardous_parts": ["CMOS Battery", "Backup Battery"],
        },
        "phone": {
            "valuable_components": ["Screen", "Battery", "Camera Module", "Storage Chip", "Logic Board"],
            "reusable_parts": ["Screen", "Camera Module", "Charger"],
            "recyclable_materials": ["Glass", "Aluminum", "Copper", "Rare Metals"],
            "hazardous_parts": ["Lithium Battery"],
        },
        "av_equipment": {
            "valuable_components": ["Speaker Unit", "Camera Lens", "Microphone", "Control Board"],
            "reusable_parts": ["Cables", "Speaker Unit", "Microphone"],
            "recyclable_materials": ["Plastic", "Copper", "Magnets", "Circuit Boards"],
            "hazardous_parts": ["Small Capacitors"],
        },
        "hvac": {
            "valuable_components": ["Compressor", "Fan Motor", "Copper Coil", "Control Board"],
            "reusable_parts": ["Fan Motor", "Control Board", "Copper Coil"],
            "recyclable_materials": ["Copper", "Aluminum", "Steel", "Plastic"],
            "hazardous_parts": ["Refrigerant Gas", "Capacitors"],
        },
        "other": {
            "valuable_components": ["Circuit Board", "Power Supply", "Cables"],
            "reusable_parts": ["Cables", "Power Adapter"],
            "recyclable_materials": ["Plastic", "Copper", "Metal", "Circuit Boards"],
            "hazardous_parts": ["Capacitors"],
        },
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

    return f"The device was categorized as {category}. The model predicted '{prediction}' because " + ", ".join(reasons) + "."


# ==========================
# AI Agent Advisor
# ==========================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()


def get_openai_client():
    """Create a direct OpenAI client when OPENAI_API_KEY is configured."""
    if not OPENAI_API_KEY:
        return None

    return OpenAI(api_key=OPENAI_API_KEY)


def find_device_record(serial_number):
    """Return one device as a JSON-safe dictionary, or None when it is not found."""
    if not serial_number:
        return None

    df = load_dataset()
    if df is None or "serial_lookup" not in df.columns:
        return None

    serial = clean_serial(serial_number)
    row = df[df["serial_lookup"] == serial]
    if row.empty:
        return None

    record = row.iloc[0]

    def value(name, default=""):
        result = record.get(name, default)
        if pd.isna(result):
            return default
        return result

    return {
        "serial_number": clean_serial(value("serial_number")),
        "item_name": str(value("item_name")),
        "price": float(value("price", 0) or 0),
        "item_age_years": float(value("item_age_years", 0) or 0),
        "status": str(value("status")),
        "item_category": str(value("item_category")),
        "itad_decision": str(value("itad_decision")),
        "current_value": float(value("current_value", 0) or 0),
        "repair_cost_ratio": float(value("repair_cost_ratio", 0) or 0),
        "building": str(value("building")),
        "department": str(value("department")),
        "room": str(value("room")),
    }


def build_rule_based_advice(question, device=None, language="en"):
    """Safe fallback response when OpenAI is not configured or unavailable."""
    is_ar = str(language).lower().startswith("ar")

    if device:
        decision = device.get("itad_decision") or "Review for E-Waste"
        age = device.get("item_age_years", 0)
        status = device.get("status") or "Unknown"
        serial = device.get("serial_number") or "Unknown"
        item = device.get("item_name") or "Device"
        ratio = device.get("repair_cost_ratio", 0)

        if is_ar:
            return (
                f"تقييم الجهاز: {item}، الرقم التسلسلي {serial}.\n"
                f"التوصية الحالية: {decision}.\n"
                f"السبب: عمر الجهاز {age} سنة، حالته {status}، ونسبة تكلفة الإصلاح {ratio:.2f}.\n"
                "الخطوات المطلوبة: انسخ البيانات المهمة، نفّذ مسحًا آمنًا للبيانات، افحص الأجزاء القابلة لإعادة الاستخدام، "
                "ثم وجّه الجهاز إلى الإصلاح أو إعادة الاستخدام أو مركز تدوير معتمد حسب القرار."
            )

        return (
            f"Device assessment: {item}, serial {serial}.\n"
            f"Current recommendation: {decision}.\n"
            f"Reason: the device is {age} years old, its status is {status}, and its repair-cost ratio is {ratio:.2f}.\n"
            "Required steps: back up needed data, perform secure data wiping, inspect reusable components, "
            "then send the device for repair, reuse, or certified recycling according to the recommendation."
        )

    if is_ar:
        return (
            "أستطيع مساعدتك في قرارات ITAD، إعادة الاستخدام، الإصلاح، التبرع، إعادة البيع، مسح البيانات، "
            "والتدوير الآمن. أدخل الرقم التسلسلي للحصول على نصيحة مرتبطة ببيانات الجهاز."
        )

    return (
        "I can advise on ITAD decisions, reuse, repair, donation, resale, secure data wiping, and certified recycling. "
        "Add a serial number to receive advice grounded in the device database."
    )


def ask_openai_advisor(question, device=None, language="en"):
    client = get_openai_client()
    if client is None:
        return build_rule_based_advice(question, device, language), False

    device_context = "No matching device was supplied."
    if device:
        device_context = "\n".join(f"- {key}: {value}" for key, value in device.items())

    system_prompt = """
You are the Smart E-Waste ITAD Advisor for a university asset-management system.
Use only the supplied device record and general e-waste knowledge. Never invent a database fact.
Recommend one practical action: Keep in Use, Maintenance Check, Repair, Refurbish, Resell, Donate,
Review for E-Waste, Recycle / Dispose, or Secure Disposal.
Always mention secure data wiping before transfer, resale, donation, recycling, or disposal.
Never recommend placing electronics, batteries, toner, refrigerants, or circuit boards in ordinary trash.
Keep the response concise, professional, and actionable. Reply in Arabic when language is ar; otherwise reply in English.
When a device is supplied, use these headings: Assessment, Recommended action, Reason, Required steps, Environmental note.
""".strip()

    user_prompt = f"Language: {language}\nUser question: {question}\nDevice record:\n{device_context}"

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=system_prompt,
        input=user_prompt,
        max_output_tokens=500,
    )

    answer = getattr(response, "output_text", "") or ""
    if not answer.strip():
        return build_rule_based_advice(question, device, language), False

    return answer.strip(), True

# ==========================
# Routes
# ==========================

@app.route("/")
def home():
    return "Smart E-Waste ITAD API is running"


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
            "repair_cost_ratio": repair_ratio,
        }])

        sample = sample[features]

        prediction_code = model.predict(sample)[0]
        prediction_label = encoders["itad_decision"].classes_[prediction_code]

        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)[0]
            confidence = round(float(max(probabilities)) * 100, 2)

        components = component_analysis(category, condition, age, repair_ratio)
        explanation = generate_explanation(prediction_label, category, age, default_life, warranty, repair_ratio)

        return jsonify({
            "prediction": prediction_label,
            "confidence": confidence,
            "item_category": category,
            "current_value": round(current_value, 2),
            "estimated_repair_cost": round(repair_cost, 2),
            "repair_cost_ratio": round(repair_ratio, 2),
            "default_life_years": default_life,
            "warranty_years": warranty,
            "out_of_warranty": out_of_warranty,
            "exceeded_life": exceeded_life,
            "components": components,
            "explanation": explanation,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/advisor", methods=["POST"])
def advisor():
    try:
        data = request.get_json(silent=True) or {}
        question = str(data.get("question", "")).strip()
        serial_number = clean_serial(data.get("serial_number", ""))
        language = str(data.get("language", "en")).lower()

        if not question:
            return jsonify({"error": "Question is required"}), 400
        if len(question) > 2000:
            return jsonify({"error": "Question is too long"}), 400

        device = find_device_record(serial_number) if serial_number else None

        try:
            answer, ai_used = ask_openai_advisor(question, device, language)
        except Exception as ai_error:
            app.logger.exception("OpenAI advisor failed: %s", ai_error)
            answer = build_rule_based_advice(question, device, language)
            ai_used = False

        return jsonify({
            "answer": answer,
            "device_found": device is not None,
            "device": device,
            "serial_number": serial_number or None,
            "ai_used": ai_used,
        })

    except Exception as e:
        app.logger.exception("Advisor endpoint failed: %s", e)
        return jsonify({"error": "The advisor is currently unavailable"}), 500


@app.route("/statistics", methods=["GET"])
def statistics():
    df = load_dataset()

    if df is not None and "itad_decision" in df.columns and "item_category" in df.columns:
        total_devices = int(len(df))
        ewaste_devices = int(df.get("is_ewaste", pd.Series(dtype=int)).fillna(0).sum()) if "is_ewaste" in df.columns else 0
        avg_value = float(df["current_value"].fillna(0).mean()) if "current_value" in df.columns else 0
        predictions = {str(k): int(v) for k, v in df["itad_decision"].fillna("Unknown").value_counts().to_dict().items()}
        categories = {str(k): int(v) for k, v in df["item_category"].fillna("other").value_counts().to_dict().items()}
    else:
        total_devices = 10668
        ewaste_devices = 2042
        avg_value = 71.10
        predictions = {
            "Review for E-Waste": 7377,
            "Recycle / Dispose": 2041,
            "Maintenance Check": 729,
            "Keep in Use": 520,
            "Repair": 1,
        }
        categories = {
            "other": 4084,
            "computer": 2344,
            "display": 1993,
            "hvac": 1018,
            "av_equipment": 610,
            "network": 301,
            "printer": 296,
            "phone": 22,
        }

    return jsonify({
        "total_devices": total_devices,
        "ewaste_devices": ewaste_devices,
        "average_current_value": round(avg_value, 2),
        "model_accuracy": 99.84,
        "predictions": predictions,
        "categories": categories,
        "environmental_impact": {
            "co2_saved_kg": 3456,
            "materials_recovered_kg": 2789,
            "hazardous_waste_handled_kg": 850,
        },
    })


@app.route("/device/<serial_number>", methods=["GET"])
def get_device(serial_number):
    try:
        df = load_dataset()
        if df is None:
            return jsonify({"error": "Dataset file not found on server"}), 404

        if "serial_lookup" not in df.columns:
            df["serial_lookup"] = df["serial_number"].astype(str).str.replace(".0", "", regex=False).str.strip()

        serial = clean_serial(serial_number)
        row = df[df["serial_lookup"] == serial]

        if row.empty:
            return jsonify({"error": "Device not found"}), 404

        device = row.iloc[0]

        return jsonify({
            "serial_number": clean_serial(device.get("serial_number", "")),
            "item_name": str(device.get("item_name", "")),
            "price": float(device.get("price", 0) or 0),
            "item_age_years": float(device.get("item_age_years", 0) or 0),
            "status": str(device.get("status", "")),
            "item_category": str(device.get("item_category", "")),
            "itad_decision": str(device.get("itad_decision", "")),
            "current_value": float(device.get("current_value", 0) or 0),
            "repair_cost_ratio": float(device.get("repair_cost_ratio", 0) or 0),
            "building": str(device.get("building", "")),
            "department": str(device.get("department", "")),
            "room": str(device.get("room", "")),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
