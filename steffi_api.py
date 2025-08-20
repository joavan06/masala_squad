from flask import Flask, request, jsonify
import mysql.connector
import json
from transformers import pipeline

# ---------------- AI MODEL CONFIGURATION ----------------
CANDIDATE_LABELS = [
    "cardiovascular",
    "respiratory",
    "metabolic",
    "infectious",
    "neurological",
    "dermatological",
    "musculoskeletal",
    "gastrointestinal",
    "psychiatric",
    "other"
]

MODEL_NAME = "facebook/bart-large-mnli"
classifier = pipeline("zero-shot-classification", model=MODEL_NAME)


def build_prompt(personal_info: dict, medical_info: dict) -> str:
    """Builds descriptive prompt from personal and medical info."""
    parts = []

    # Personal Info
    parts.append("PERSONAL INFO:")
    for k, v in [
        ("Name", personal_info.get("full_name")),
        ("DOB", personal_info.get("date_of_birth")),
        ("Address", personal_info.get("address")),
        ("Contact", personal_info.get("contact_number")),
        ("Emergency Contact", personal_info.get("emergency_contact"))
    ]:
        if v:
            parts.append(f"{k}: {v}")

    # Medical Info
    parts.append("\nMEDICAL INFO:")
    for k, v in [
        ("Gender", medical_info.get("gender")),
        ("Age", medical_info.get("age")),
        ("Blood group", medical_info.get("blood_group")),
        ("Past illnesses", medical_info.get("past_illnesses")),
        ("Current conditions", medical_info.get("current_medical_conditions")),
        ("Allergies", medical_info.get("allergies")),
        ("Medications", medical_info.get("current_medications")),
        ("Past surgeries / injuries", medical_info.get("past_surgeries_major_injuries")),
        ("Family history", medical_info.get("family_medical_history")),
        ("Lifestyle", medical_info.get("lifestyle_factors"))
    ]:
        if v:
            parts.append(f"{k}: {v}")

    return " | ".join(parts)[:4000]  # limit length


def predict_generic_category(personal_info: dict, medical_info: dict, top_k: int = 3):
    """Runs zero-shot classification and returns predictions."""
    text = build_prompt(personal_info, medical_info)
    result = classifier(text, candidate_labels=CANDIDATE_LABELS, multi_label=False)

    labels = result["labels"][:top_k]
    scores = result["scores"][:top_k]
    predictions = [{"label": l, "score": float(s)} for l, s in zip(labels, scores)]

    return {
        "input_text": text,
        "predictions": predictions,
        "note": (
            "This is an automated categorization suggestion (not a medical diagnosis). "
            "Consult a healthcare professional for confirmation."
        )
    }

# ---------------- FLASK APP ----------------
app = Flask(__name__)

@app.route("/api/predict_disease", methods=["POST"])
def predict_disease():
    """Fetch user data from DB by s_no, then run AI prediction."""
    try:
        data = request.get_json()
        if not data or "s_no" not in data:
            return jsonify({"error": "Missing 's_no' in request body"}), 400

        s_no = data["s_no"]

        # Connect to DB
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",  # change if needed
            database="masala_squad"
        )
        cursor = conn.cursor(dictionary=True)

        # Fetch user
        cursor.execute("SELECT * FROM user_profile WHERE s_no = %s", (s_no,))
        record = cursor.fetchone()

        if not record:
            return jsonify({"error": "User not found"}), 404

        # Format date_of_birth
        if record.get("date_of_birth"):
            record["date_of_birth"] = record["date_of_birth"].strftime("%d %B %Y")

        personal_info = {
            "s_no": record["s_no"],
            "full_name": record["full_name"],
            "date_of_birth": record["date_of_birth"],
            "address": record["address"],
            "contact_number": record["contact_number"],
            "emergency_contact": record["emergency_contact"]
        }

        medical_info = {
            "gender": record["gender"],
            "age": record["age"],
            "blood_group": record["blood_group"],
            "past_illnesses": record["past_illnesses"],
            "current_medical_conditions": record["current_medical_conditions"],
            "allergies": record["allergies"],
            "current_medications": record["current_medications"],
            "past_surgeries_major_injuries": record["past_surgeries_major_injuries"],
            "family_medical_history": record["family_medical_history"],
            "lifestyle_factors": record["lifestyle_factors"]
        }

        # AI prediction
        ai_result = predict_generic_category(personal_info, medical_info, top_k=3)

        return jsonify({
            "personal_info": personal_info,
            "medical_info": medical_info,
            "ai_prediction": ai_result
        })

    except mysql.connector.Error as err:
        return jsonify({"error": str(err)}), 500

    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    app.run(debug=True)
