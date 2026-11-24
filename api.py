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
# 2. CLASSIFICATION (Gibson) -- employee attrition
# ---------------------------------------------------------------
@app.route("/api/v1/models/employee-attrition-classifier/predictions", methods=["POST"])
def predict_attrition():
    return jsonify({"error": "Not implemented yet"}), 501


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