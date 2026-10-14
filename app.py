import os
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load saved SVM model and scaler
model = joblib.load("svm_churn_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        tenure = float(data["tenure"])
        monthly_charges = float(data["monthly_charges"])
        total_charges = float(data["total_charges"])
        contract_type = float(data["contract_type"])

        # Structure input for model prediction
        features = np.array([[tenure, monthly_charges, total_charges, contract_type]])
        scaled_features = scaler.transform(features)

        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]

        return jsonify(
            {
                "churn_prediction": int(prediction),
                "risk_label": "High Risk" if prediction == 1 else "Low Risk",
                "churn_probability": round(float(probability) * 100, 2),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
