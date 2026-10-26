import os
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 1. Load the best saved model and scaler
MODEL_PATH = "best_churn_model.pkl"  # Saved from your evaluation script
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Models like SVC require feature scaling; Tree-based models (RandomForest, DecisionTree) do not
REQUIRES_SCALING = hasattr(model, "kernel") or "SVC" in type(model).__name__


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validate required input keys
        required_fields = ["tenure", "monthly_charges", "total_charges", "contract_type"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: '{field}'"}), 400

        tenure = float(data["tenure"])
        monthly_charges = float(data["monthly_charges"])
        total_charges = float(data["total_charges"])
        contract_type = float(data["contract_type"])

        # 2. Structure feature array matching training column order:
        # [tenure, MonthlyCharges, TotalCharges, Contract_Numeric]
        raw_features = np.array([[tenure, monthly_charges, total_charges, contract_type]])

        # 3. Scale features ONLY if model relies on distance metrics (e.g., SVM)
        features_to_predict = scaler.transform(raw_features) if REQUIRES_SCALING else raw_features

        # 4. Predict class and probability
        prediction = int(model.predict(features_to_predict)[0])
        
        # Check if model supports probability estimation
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(features_to_predict)[0][1])
        else:
            probability = float(prediction)

        return jsonify(
            {
                "churn_prediction": prediction,
                "risk_label": "High Risk" if prediction == 1 else "Low Risk",
                "churn_probability": round(probability * 100, 2),
            }
        )

    except ValueError:
        return jsonify({"error": "Invalid numerical input provided."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
