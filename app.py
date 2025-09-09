from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import json

app = Flask(__name__)

# Load pipeline (preprocessor + model)
model = joblib.load("unified_fraud_model.pkl")

# Load decision threshold
with open("threshold.json", "r") as f:
    threshold = json.load(f)["decision_threshold"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract inputs
        amount = float(data["amount"])
        hour = int(data["hour"])
        source_type = data["source_type"]

        # Build DataFrame for pipeline
        X = pd.DataFrame([{
            "amount": amount,
            "hour": hour,
            "source_type": source_type
        }])

        # Predict fraud probability
        proba = model.predict_proba(X)[0, 1]

        # Apply threshold
        prediction = 1 if proba >= threshold else 0

        return jsonify({
            "proba": round(float(proba), 4),
            "threshold": threshold,
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
