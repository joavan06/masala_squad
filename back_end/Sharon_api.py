from flask import Flask, request, jsonify
import mysql.connector
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

# ---------------- FLASK APP ----------------
app = Flask(__name__)

# Function to connect MySQL
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="hospital"
    )

# Predict sickness category
def predict_category(symptoms: str, top_k: int = 1):
    result = classifier(symptoms, candidate_labels=CANDIDATE_LABELS, multi_label=False)
    label = result["labels"][0]
    score = float(result["scores"][0])
    return label, score

@app.route("/api/find_hospital", methods=["POST"])
def find_hospital():
    try:
        data = request.get_json()
        if not data or "symptoms" not in data:
            return jsonify({"error": "Missing 'symptoms' in request body"}), 400

        symptoms = data["symptoms"]

        # Step 1: Predict category
        predicted_label, confidence = predict_category(symptoms)

        # Step 2: Map AI label to hospital specialization keywords
        specialization_map = {
            "cardiovascular": "Cardiology",
            "respiratory": "Pulmonology",
            "metabolic": "Endocrinology",
            "infectious": "Infectious",
            "neurological": "Neurology",
            "dermatological": "Dermatology",
            "musculoskeletal": "Orthopedics",
            "gastrointestinal": "Gastroenterology",
            "psychiatric": "Psychiatry",
            "other": "General"
        }

        specialization_keyword = specialization_map.get(predicted_label, "General")

        # Step 3: Search hospitals in MySQL with matching specialization
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        query = "SELECT * FROM hospitals WHERE specializations LIKE %s"
        cursor.execute(query, (f"%{specialization_keyword}%",))
        hospitals = cursor.fetchall()

        cursor.close()
        conn.close()

        return jsonify({
            "symptoms": symptoms,
            "predicted_category": predicted_label,
            "confidence": confidence,
            "matched_specialization": specialization_keyword,
            "recommended_hospitals": hospitals
        })

    except mysql.connector.Error as err:
        return jsonify({"error": str(err)}), 500


if __name__ == "__main__":
    app.run(debug=True)
