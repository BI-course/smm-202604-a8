import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# ---------------- Classification (Gibson) ----------------
CLASSIFIER_PATH = "model/svc_classifier_for_sme_credit_risk.joblib"

CLASSIFIER_FEATURES = [
    "sector", "county", "quarter", "loan_purpose", "owner_education_level",
    "business_age_years", "num_employees", "monthly_revenue_kes",
    "loan_amount_kes", "loan_term_months", "interest_rate_pct",
    "collateral_value_kes", "debt_to_income_ratio", "credit_score",
    "num_previous_loans", "missed_payments_count", "online_presence_score",
    "customer_satisfaction_score", "competitor_density",
]

# Loaded once at startup. Never call .fit() here.
classifier = joblib.load(CLASSIFIER_PATH)


@app.route("/predict/classification", methods=["POST"])
def predict_classification():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    missing = [f for f in CLASSIFIER_FEATURES if f not in data]
    if missing:
        return jsonify({"error": "Missing required fields", "missing": missing}), 400

    try:
        row = pd.DataFrame([[data[f] for f in CLASSIFIER_FEATURES]],
                           columns=CLASSIFIER_FEATURES)
        pred = classifier.predict(row)[0]
        result = {"prediction": pred.item() if hasattr(pred, "item") else pred}

        if hasattr(classifier, "predict_proba"):
            try:
                probs = classifier.predict_proba(row)[0]
                result["probabilities"] = {
                    str(c): float(p) for c, p in zip(classifier.classes_, probs)
                }
            except Exception:
                pass  # SVC trained without probability=True

        return jsonify(result), 200
    except (ValueError, TypeError) as e:
        return jsonify({"error": "Invalid input values", "detail": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Prediction failed", "detail": str(e)}), 500


# ---------------- Other members' routes go ABOVE the run block ----------------

if __name__ == "__main__":
    app.run(debug=True)