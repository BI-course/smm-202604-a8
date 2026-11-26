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
# 2. CLASSIFICATION (Gibson) -- SME credit risk
# ---------------------------------------------------------------

classification_model = joblib.load(
    os.path.join(MODEL_DIR, "svc_classifier_for_sme_credit_risk.joblib")
)

CLASSIFICATION_FEATURES = [
    "sector", "county", "quarter", "loan_purpose",
    "owner_education_level", "business_age_years",
    "num_employees", "monthly_revenue_kes",
    "loan_amount_kes", "loan_term_months",
    "interest_rate_pct", "collateral_value_kes",
    "debt_to_income_ratio", "credit_score",
    "num_previous_loans", "missed_payments_count",
    "online_presence_score", "customer_satisfaction_score",
    "competitor_density",
]


@app.route(
    "/api/v1/models/sme-credit-risk-classifier/predictions",
    methods=["POST"]
)
def predict_credit_risk():

    payload = request.get_json(silent=True)

    if not payload:
        return jsonify({"error": "Send a JSON body"}), 400

    missing = [
        feature for feature in CLASSIFICATION_FEATURES
        if feature not in payload
    ]

    if missing:
        return jsonify({
            "error": "Missing inputs",
            "missing": missing
        }), 400

    try:
        input_row = pd.DataFrame(
            [payload],
            columns=CLASSIFICATION_FEATURES
        )

        pred = classification_model.predict(input_row)[0]

        result = {
            "prediction": (
                pred.item() if hasattr(pred, "item") else pred
            ),
            "model": "SVC Classifier - SME Credit Risk"
        }

        if hasattr(classification_model, "predict_proba"):
            probs = classification_model.predict_proba(input_row)[0]

            result["probabilities"] = {
                str(c): float(p)
                for c, p in zip(
                    classification_model.classes_, probs
                )
            }

        return jsonify(result), 200

    except (ValueError, TypeError) as e:
        return jsonify({
            "error": "Invalid input values",
            "detail": str(e)
        }), 400

# ---------------------------------------------------------------
# 3. CLUSTERING (Marwa) -- retail stores
# ---------------------------------------------------------------
# Load the saved package once when the server starts (never retrain here).
# The notebook saved a dict: {"scaler": StandardScaler, "kmeans": KMeans (k=4), "features": [...]}
clustering_artifact = joblib.load(
    os.path.join(MODEL_DIR, "retail_store_kmeans_model.joblib")
)
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]

# The exact column names (and order) the model was trained on
CLUSTERING_FEATURES = clustering_artifact["features"]
# = ["monthly_revenue_kes", "daily_foot_traffic", "staff_count", "avg_basket_size_kes"]

# Segment names from the notebook's cluster-profiling section
CLUSTER_NAMES = {
    0: "High-Traffic Mid-Performing Stores",
    1: "High-Performance Large-Scale Stores",
    2: "High-Basket, Low-Traffic Stores",
    3: "Low-Performance, Low-Traffic Stores",
}


@app.route("/api/v1/models/retail-store-clusterer/predictions", methods=["POST"])
def predict_store_cluster():
    # Read the JSON body (silent=True gives None instead of crashing on bad JSON)
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Send a JSON body"}), 400

    # Basic error handling: tell the user which inputs are missing
    missing = [f for f in CLUSTERING_FEATURES if f not in payload or payload[f] in (None, "")]
    if missing:
        return jsonify({"error": "Missing inputs", "missing": missing}), 400

    # K-Means works on numbers only, so reject text values clearly
    try:
        input_row = pd.DataFrame([payload])[CLUSTERING_FEATURES].astype(float)
    except (ValueError, TypeError):
        return jsonify({
            "error": "All inputs must be numbers",
            "expected_features": CLUSTERING_FEATURES,
        }), 400

    # Apply the SAME fitted scaler from training first, then predict the cluster
    scaled_row = clustering_scaler.transform(input_row)
    cluster = int(clustering_kmeans.predict(scaled_row)[0])

    return jsonify({
        "cluster": cluster,
        "cluster_name": CLUSTER_NAMES.get(cluster, f"Cluster {cluster}"),
        "model": "K-Means Clustering (k=4) - Retail Stores",
    }), 200



# ---------------------------------------------------------------
# 4. RECOMMENDER (David Usyu Mutinda, 169382) -- music association rules
# ---------------------------------------------------------------
# Load the persisted music rules once. Mining lives in scripts/build_music_rules.py.
recommender_artifact = joblib.load(
    os.path.join(MODEL_DIR, "music_track_recommender.joblib")
)
recommendation_lookup = recommender_artifact["recommendation_lookup"]
track_id_to_name = recommender_artifact["track_id_to_name"]


@app.route("/api/v1/models/music-track-recommender/predictions", methods=["POST"])
def recommend_tracks():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object with a track_ids list"}), 400

    track_ids = payload.get("track_ids")
    if not isinstance(track_ids, list) or not track_ids:
        return jsonify({"error": "track_ids must be a non-empty list of track IDs"}), 400
    if any(not isinstance(track, str) or not track.strip() for track in track_ids):
        return jsonify({"error": "Each track ID must be a non-empty string"}), 400

    top_n = payload.get("top_n", 5)
    if type(top_n) is not int or not 1 <= top_n <= 20:
        return jsonify({"error": "top_n must be an integer between 1 and 20"}), 400

    played = {track.strip() for track in track_ids}
    unknown = sorted(played.difference(track_id_to_name))
    if unknown:
        return jsonify({"error": "Unknown track IDs", "unknown_track_ids": unknown}), 400

    # Match full antecedents; exclude played tracks; keep the strongest
    # confidence/lift/support evidence for each unseen track.
    best = {}
    for antecedent, candidates in recommendation_lookup.items():
        if antecedent.issubset(played):
            for track, confidence, lift, support in candidates:
                score = (float(confidence), float(lift), float(support))
                if track not in played and (track not in best or score > best[track]):
                    best[track] = score

    ranked = sorted(best.items(), key=lambda item: tuple(-v for v in item[1]) + (item[0],))[:top_n]
    return jsonify({
        "track_ids": sorted(played),
        "recommendations": [
            {
                "track_id": track,
                "track_name": track_id_to_name[track],
                "confidence": round(confidence, 4),
                "lift": round(lift, 4),
                "support": round(support, 4),
            }
            for track, (confidence, lift, support) in ranked
        ],
        "model": "Music track association rules (Apriori)",
    }), 200


if __name__ == "__main__":
    app.run(debug=True)
