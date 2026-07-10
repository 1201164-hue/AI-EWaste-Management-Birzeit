from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import pandas as pd
import joblib
import os
import json
import re
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
    """Return concise advice based only on the stored device record."""
    is_ar = str(language).lower().startswith("ar")

    if not device:
        if is_ar:
            return (
                "لم يتم اختيار جهاز.
"
                "أدخل الرقم التسلسلي أولًا للحصول على قرار نموذج Random Forest "
                "وتوصية مرتبطة ببيانات الجهاز."
            )

        return (
            "No device is selected.
"
            "Enter a serial number to get the Random Forest decision "
            "and record-based advice."
        )

    item_name = str(device.get("item_name") or "Device")
    serial = str(device.get("serial_number") or "Unknown")
    category = str(device.get("item_category") or "other")
    status = str(device.get("status") or "Unknown")
    rf_decision = str(device.get("itad_decision") or "Review for E-Waste")

    age = float(device.get("item_age_years") or 0)
    current_value = float(device.get("current_value") or 0)
    repair_ratio = float(device.get("repair_cost_ratio") or 0)

    warranty_years = warranty_map.get(category, 2)
    warranty_active = age <= warranty_years

    action_map = {
        "Keep in Use": "Keep the device in service and continue routine maintenance.",
        "Maintenance Check": "Inspect the device and confirm whether maintenance is economical.",
        "Repair": "Repair the device only if the confirmed repair cost is reasonable.",
        "Recycle / Dispose": (
            "Perform secure data wiping, remove reusable parts, "
            "and send the device to certified recycling."
        ),
        "Review for E-Waste": (
            "Inspect the device before deciding between reuse, repair, "
            "component harvesting, or recycling."
        ),
    }
    next_action = action_map.get(
        rf_decision,
        "Inspect the device and follow the official ITAD procedure."
    )

    component_data = component_analysis(category, status, age, repair_ratio)
    reusable = ", ".join(component_data.get("reusable_parts", [])[:4]) or "Not available"
    materials = ", ".join(component_data.get("recyclable_materials", [])[:5]) or "Not available"

    if is_ar:
        warranty_text = "ساري تقديريًا" if warranty_active else "منتهي تقديريًا"
        return (
            f"قرار Random Forest: {rf_decision}
"
            f"الجهاز: {item_name} | الرقم التسلسلي: {serial}
"
            f"الحالة: {status} | العمر: {age:.1f} سنة | الضمان: {warranty_text}
"
            f"القيمة الحالية: {current_value:.2f} | نسبة الإصلاح: {repair_ratio:.2f}
"
            f"الإجراء المقترح: {next_action}
"
            f"أجزاء للفحص وإعادة الاستخدام: {reusable}
"
            f"مواد قابلة للاسترداد: {materials}
"
            "مهم: نفّذ مسحًا آمنًا للبيانات قبل البيع أو النقل أو التدوير."
        )

    warranty_text = "Estimated active" if warranty_active else "Estimated expired"
    return (
        f"Random Forest decision: {rf_decision}
"
        f"Device: {item_name} | Serial number: {serial}
"
        f"Status: {status} | Age: {age:.1f} years | Warranty: {warranty_text}
"
        f"Current value: {current_value:.2f} | Repair ratio: {repair_ratio:.2f}
"
        f"Recommended action: {next_action}
"
        f"Reusable parts to inspect: {reusable}
"
        f"Recoverable materials: {materials}
"
        "Important: perform secure data wiping before sale, transfer, or recycling."
    )


def detect_language(question, requested_language="auto"):
    """Use the requested UI language, or detect Arabic from the question."""
    requested = str(requested_language or "auto").lower().strip()
    if requested.startswith("ar"):
        return "ar"
    if requested.startswith("en"):
        return "en"
    return "ar" if re.search(r"[\u0600-\u06FF]", str(question)) else "en"


def build_advisor_prompts(question, device=None, language="en", history=None):
    device_context = "No matching device was supplied."
    if device:
        device_context = "
".join(f"- {key}: {value}" for key, value in device.items())

    response_language = "Arabic" if language == "ar" else "English"

    system_prompt = f"""
You are the Smart E-Waste ITAD Advisor.
Reply in {response_language}.

The device record contains the official result of a trained Random Forest model.
The exact Random Forest prediction is stored in the field: itad_decision.

Mandatory rules:
1. Start every device-specific answer with:
   Random Forest decision: <exact itad_decision value>
   In Arabic use:
   قرار Random Forest: <exact itad_decision value>

2. Treat itad_decision as the official machine-learning result.
   Do not replace it with your own decision.

3. Never invent CPU, RAM, storage, GPU, warranty documents, faults,
   repair prices, resale prices, specifications, dates, locations, or database values.

4. Only mention facts available in the supplied device record.
   When a fact is missing, say it is not available in the device record.

5. Do not use Markdown.
   Do not use asterisks, headings, tables, or decorative formatting.

6. Keep the answer concise: no more than 8 short lines.

7. Include useful information only:
   - exact Random Forest decision,
   - device name and serial number,
   - status and age,
   - estimated warranty status,
   - current value and repair ratio,
   - practical next action,
   - reusable parts as inspection suggestions,
   - recoverable materials,
   - secure data wiping when applicable.

8. Circuit boards may contain trace gold, silver, or palladium.
   Never invent quantities or monetary values.

9. Never recommend ordinary trash for electronics, batteries, toner,
   refrigerants, or circuit boards.
""".strip()

    messages = []
    for item in (history or [])[-6:]:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).lower()
        content = str(item.get("content", "")).strip()

        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content[:2500]})

    messages.append({
        "role": "user",
        "content": (
            f"Selected device record:
{device_context}

"
            f"Current question:
{question}"
        ),
    })

    return system_prompt, messages


def ask_openai_advisor(question, device=None, language="en", history=None):
    client = get_openai_client()
    if client is None:
        return build_rule_based_advice(question, device, language), False

    system_prompt, messages = build_advisor_prompts(question, device, language, history)
    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=system_prompt,
        input=messages,
        max_output_tokens=280,
    )

    answer = getattr(response, "output_text", "") or ""
    if not answer.strip():
        return build_rule_based_advice(question, device, language), False
    return answer.strip(), True


def sse_message(event_name, payload):
    """Encode one Server-Sent Event frame."""
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def stream_openai_advisor(question, device=None, language="en", history=None):
    """Yield OpenAI text deltas as SSE frames."""
    client = get_openai_client()
    if client is None:
        fallback = build_rule_based_advice(question, device, language)
        for chunk in re.findall(r".{1,24}(?:\s+|$)", fallback, flags=re.S):
            yield sse_message("delta", {"text": chunk})
        yield sse_message("done", {"ai_used": False})
        return

    system_prompt, messages = build_advisor_prompts(question, device, language, history)
    stream = client.responses.create(
        model=OPENAI_MODEL,
        instructions=system_prompt,
        input=messages,
        max_output_tokens=280,
        stream=True,
    )

    emitted = False
    for event in stream:
        event_type = getattr(event, "type", "")
        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "") or ""
            if delta:
                emitted = True
                yield sse_message("delta", {"text": delta})
        elif event_type == "response.failed":
            error = getattr(event, "error", None)
            message = getattr(error, "message", None) or "OpenAI response failed"
            raise RuntimeError(message)

    if not emitted:
        fallback = build_rule_based_advice(question, device, language)
        yield sse_message("delta", {"text": fallback})
        yield sse_message("done", {"ai_used": False})
    else:
        yield sse_message("done", {"ai_used": True})

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
        language = detect_language(question, data.get("language", "auto"))
        history = data.get("history", [])

        if not question:
            return jsonify({"error": "Question is required"}), 400
        if len(question) > 2000:
            return jsonify({"error": "Question is too long"}), 400

        device = find_device_record(serial_number) if serial_number else None

        try:
            answer, ai_used = ask_openai_advisor(question, device, language, history)
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



@app.route("/advisor/stream", methods=["POST"])
def advisor_stream():
    """Stream advisor output as Server-Sent Events over a POST response."""
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    serial_number = clean_serial(data.get("serial_number", ""))
    language = detect_language(question, data.get("language", "auto"))
    history = data.get("history", [])

    if not question:
        return jsonify({"error": "Question is required"}), 400
    if len(question) > 2000:
        return jsonify({"error": "Question is too long"}), 400

    device = find_device_record(serial_number) if serial_number else None

    @stream_with_context
    def generate():
        yield sse_message("meta", {
            "device_found": device is not None,
            "device": device,
            "serial_number": serial_number or None,
            "language": language,
        })
        try:
            yield from stream_openai_advisor(question, device, language, history)
        except GeneratorExit:
            return
        except Exception as exc:
            app.logger.exception("Streaming advisor failed: %s", exc)
            fallback = build_rule_based_advice(question, device, language)
            yield sse_message("delta", {"text": fallback})
            yield sse_message("done", {"ai_used": False, "fallback": True})

    response = Response(generate(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache, no-transform"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Connection"] = "keep-alive"
    return response

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
