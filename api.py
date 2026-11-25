"""
api.py -- Flask API serving our group's four saved models.
Models are loaded from disk once at startup. Nothing is retrained here.
"""
import os


import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # fine for the demo; not for production

MODEL_DIR = "./model"


# ---------------------------------------------------------------
# 1. REGRESSION (Yeshua) -- rental price
# ---------------------------------------------------------------
# Load the saved pipeline once when the server starts
regression_model = joblib.load(
    os.path.join(MODEL_DIR, "20261124_171555_ridge_regressor_for_rental_price.joblib")
)

# The exact column names the model was trained on
REGRESSION_FEATURES = [
    "size_sqm", "num_bedrooms", "distance_to_cbd_km", "garage_size_sqm",
    "security_rating", "amenities_score", "nearby_schools_count",
    "property_age_years", "property_type", "neighborhood", "furnishing_status",
]

# Set to True ONLY if the notebook trained on a log-transformed price
TARGET_WAS_LOG_TRANSFORMED = True


@app.route("/api/v1/models/rental-price-regressor/predictions", methods=["POST"])
def predict_rental_price():
    # Read the JSON body (silent=True gives None instead of crashing on bad JSON)
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Send a JSON body"}), 400

    # Basic error handling: tell the user which inputs are missing
    missing = [f for f in REGRESSION_FEATURES if f not in payload]
    if missing:
        return jsonify({"error": "Missing inputs", "missing": missing}), 400

    # Build a one-row table in the same column order as training
    input_row = pd.DataFrame([payload])[REGRESSION_FEATURES]

    # The pipeline does the preprocessing and the prediction
    prediction = float(regression_model.predict(input_row)[0])
    if TARGET_WAS_LOG_TRANSFORMED:
        prediction = float(np.expm1(prediction))

    return jsonify({
        "predicted_rental_price": round(prediction, 2),
        "model": "Ridge Regressor",
    })


# ---------------------------------------------------------------
# 2. CLASSIFICATION (Gibson)
# ---------------------------------------------------------------
# TODO: swap in the group's attrition classifier when it is available
# (change the file name, CLASSIFICATION_FEATURES and the "model" label only)
classification_model = joblib.load(
    os.path.join(MODEL_DIR, "svc_classifier_for_sme_credit_risk.joblib")
)

CLASSIFICATION_FEATURES = [
    "sector", "county", "quarter", "loan_purpose", "owner_education_level",
    "business_age_years", "num_employees", "monthly_revenue_kes",
    "loan_amount_kes", "loan_term_months", "interest_rate_pct",
    "collateral_value_kes", "debt_to_income_ratio", "credit_score",
    "num_previous_loans", "missed_payments_count", "online_presence_score",
    "customer_satisfaction_score", "competitor_density",
]


@app.route("/api/v1/models/employee-attrition-classifier/predictions", methods=["POST"])
def predict_attrition():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Send a JSON body"}), 400

    missing = [f for f in CLASSIFICATION_FEATURES if f not in payload]
    if missing:
        return jsonify({"error": "Missing inputs", "missing": missing}), 400

    try:
        input_row = pd.DataFrame([payload])[CLASSIFICATION_FEATURES]
        pred = classification_model.predict(input_row)[0]
        result = {
            "prediction": pred.item() if hasattr(pred, "item") else pred,
            "model": "SVC Classifier",
        }
        if hasattr(classification_model, "predict_proba"):
            probs = classification_model.predict_proba(input_row)[0]
            result["probabilities"] = {
                str(c): float(p)
                for c, p in zip(classification_model.classes_, probs)
            }
        return jsonify(result), 200
    except (ValueError, TypeError) as e:
        return jsonify({"error": "Invalid input values", "detail": str(e)}), 400


# ---------------------------------------------------------------
# 3. CLUSTERING (David) -- retail stores
# ---------------------------------------------------------------
@app.route("/api/v1/models/retail-store-clusterer/predictions", methods=["POST"])
def predict_store_cluster():
    return jsonify({"error": "Not implemented yet"}), 501


# ---------------------------------------------------------------
# 4. RECOMMENDER (Marwa) -- music tracks
# ---------------------------------------------------------------
@app.route("/api/v1/models/music-track-recommender/predictions", methods=["POST"])
def recommend_tracks():
    return jsonify({"error": "Not implemented yet"}), 501


if __name__ == "__main__":
    app.run(debug=True)