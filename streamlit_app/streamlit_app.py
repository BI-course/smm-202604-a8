"""
streamlit_app.py -- Group 202604-A8, BBT 4206 (Marwa Mohammed, 171029)

One Streamlit app, one tab per model. It loads the SAME saved .joblib files
from the repo's model/ folder that api.py uses. Nothing is retrained here.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# model/ sits one level above this file (repo root), so build the path from
# this file's location. That works locally and on Streamlit Community Cloud.
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

# Exact filenames -- keep these identical to the ones in api.py
REGRESSION_FILE = "20261124_171555_ridge_regressor_for_rental_price.joblib"
CLASSIFICATION_FILE = "svc_classifier_for_sme_credit_risk.joblib"
CLUSTERING_FILE = "retail_store_kmeans_model.joblib"
RECOMMENDER_FILE = "music_track_recommender.joblib"         # Same saved music artifact as api.py

st.set_page_config(page_title="Group A8 -- Model Demos", page_icon="📊", layout="centered")


# -------------------------------------------------------------------------
# Load each artifact ONCE. Streamlit re-runs the script on every click, so
# @st.cache_resource stops joblib.load() reading the files again each time.
# A missing file shows a message on its own tab instead of crashing the app.
# -------------------------------------------------------------------------
@st.cache_resource
def load_artifact(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        return None, f"Model file `{filename}` is not in the model/ folder yet."
    try:
        return joblib.load(path), None
    except Exception as e:  # e.g. version mismatch
        return None, f"Could not load `{filename}`: {type(e).__name__}: {e}"


st.title("Group 202604-A8 -- Model Demos")
st.caption("BBT 4206 Business Intelligence II -- the same saved models served by our Flask API.")

tab_reg, tab_clf, tab_clu, tab_rec = st.tabs([
    "Rental Price Regressor", "SME Credit Risk Classifier",
    "Retail Store Clusterer", "Music Track Recommender",
])

# -------------------------------------------------------------------------
# 1. REGRESSION -- rental price (Ridge pipeline, log1p target)
# -------------------------------------------------------------------------
with tab_reg:
    model, err = load_artifact(REGRESSION_FILE)
    if err:
        st.error(err)
    else:
        st.subheader("Predict monthly rental price")
        with st.form("regression_form"):
            c1, c2 = st.columns(2)
            property_type = c1.selectbox("Property type",
                ["Apartment", "Bungalow", "Maisonette", "Studio", "Townhouse"])
            neighborhood = c2.selectbox("Neighbourhood",
                ["Karen", "Kilimani", "Ngong Road", "Ruaka", "South B", "Westlands"], index=1)
            furnishing_status = c1.selectbox("Furnishing status",
                ["Fully Furnished", "Semi-Furnished", "Unfurnished"], index=2)
            size_sqm = c2.number_input("Size (sqm)", min_value=1.0, value=86.5)
            num_bedrooms = c1.number_input("Bedrooms", min_value=0, value=3)
            distance_to_cbd_km = c2.number_input("Distance to CBD (km)", min_value=0.0, value=7.3)
            garage_size_sqm = c1.number_input("Garage size (sqm)", min_value=0.0, value=15.5)
            security_rating = c2.number_input("Security rating", min_value=0.0, value=4.0)
            amenities_score = c1.number_input("Amenities score", min_value=0.0, value=54.65)
            nearby_schools_count = c2.number_input("Nearby schools", min_value=0, value=2)
            property_age_years = c1.number_input("Property age (years)", min_value=0.0, value=6.0)
            go = st.form_submit_button("Predict rent")

        if go:
            row = pd.DataFrame([{
                "size_sqm": size_sqm, "num_bedrooms": num_bedrooms,
                "distance_to_cbd_km": distance_to_cbd_km, "garage_size_sqm": garage_size_sqm,
                "security_rating": security_rating, "amenities_score": amenities_score,
                "nearby_schools_count": nearby_schools_count,
                "property_age_years": property_age_years, "property_type": property_type,
                "neighborhood": neighborhood, "furnishing_status": furnishing_status,
            }])
            # Model was trained on log1p(price), so convert back with expm1
            price = float(np.expm1(model.predict(row)[0]))
            st.success(f"Predicted rental price: KES {price:,.2f}")

# -------------------------------------------------------------------------
# 2. CLASSIFICATION -- SME credit risk (imblearn pipeline with SVC)
# -------------------------------------------------------------------------
with tab_clf:
    model, err = load_artifact(CLASSIFICATION_FILE)
    if err:
        st.error(err)
    else:
        st.subheader("Predict SME credit risk")
        with st.form("classification_form"):
            c1, c2 = st.columns(2)
            sector = c1.selectbox("Sector", ["Agriculture", "Hospitality", "Manufacturing",
                                             "Retail", "Services", "Technology"], index=3)
            county = c2.selectbox("County", ["Eldoret", "Kisumu", "Mombasa", "Nairobi",
                                             "Nakuru", "Other"], index=3)
            quarter = c1.selectbox("Quarter", ["Q1", "Q2", "Q3", "Q4"])
            loan_purpose = c2.selectbox("Loan purpose", ["Business Expansion", "Equipment Purchase",
                                                         "Inventory Financing", "Working Capital"], index=3)
            owner_education_level = c1.selectbox("Owner education level",
                ["Primary", "Secondary", "Certificate", "Diploma", "Bachelors", "Masters", "Doctorate"],
                index=1)
            business_age_years = c2.number_input("Business age (years)", min_value=0.0, value=5.2)
            num_employees = c1.number_input("Employees", min_value=0, value=8)
            monthly_revenue_kes = c2.number_input("Monthly revenue (KES)", min_value=0.0, value=1751155.0)
            loan_amount_kes = c1.number_input("Loan amount (KES)", min_value=0.0, value=3334734.0)
            loan_term_months = c2.number_input("Loan term (months)", min_value=1, value=24)
            interest_rate_pct = c1.number_input("Interest rate (%)", min_value=0.0, value=15.82)
            collateral_value_kes = c2.number_input("Collateral value (KES)", min_value=0.0, value=3160008.0)
            debt_to_income_ratio = c1.number_input("Debt-to-income ratio", min_value=0.0, value=0.392)
            credit_score = c2.number_input("Credit score", min_value=0, value=639)
            num_previous_loans = c1.number_input("Previous loans", min_value=0, value=2)
            missed_payments_count = c2.number_input("Missed payments", min_value=0, value=0)
            online_presence_score = c1.number_input("Online presence score", min_value=0.0, value=52.8)
            customer_satisfaction_score = c2.number_input("Customer satisfaction score",
                                                          min_value=0.0, value=3.6)
            competitor_density = c1.number_input("Competitor density", min_value=0, value=6)
            go = st.form_submit_button("Predict credit risk")

        if go:
            row = pd.DataFrame([{
                "sector": sector, "county": county, "quarter": quarter,
                "loan_purpose": loan_purpose, "owner_education_level": owner_education_level,
                "business_age_years": business_age_years, "num_employees": num_employees,
                "monthly_revenue_kes": monthly_revenue_kes, "loan_amount_kes": loan_amount_kes,
                "loan_term_months": loan_term_months, "interest_rate_pct": interest_rate_pct,
                "collateral_value_kes": collateral_value_kes,
                "debt_to_income_ratio": debt_to_income_ratio, "credit_score": credit_score,
                "num_previous_loans": num_previous_loans,
                "missed_payments_count": missed_payments_count,
                "online_presence_score": online_presence_score,
                "customer_satisfaction_score": customer_satisfaction_score,
                "competitor_density": competitor_density,
            }])
            pred = model.predict(row)[0]
            st.success(f"Predicted credit risk: **{pred}**")
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(row)[0]
                st.bar_chart(pd.Series(probs, index=[str(c) for c in model.classes_], name="Probability"))

# -------------------------------------------------------------------------
# 3. CLUSTERING -- retail stores. StandardScaler + K-Means (k=4).
# -------------------------------------------------------------------------
CLUSTER_NAMES = {
    0: "High-Traffic Mid-Performing Stores",
    1: "High-Performance Large-Scale Stores",
    2: "High-Basket, Low-Traffic Stores",
    3: "Low-Performance, Low-Traffic Stores",
}

with tab_clu:
    artifact, err = load_artifact(CLUSTERING_FILE)
    if err:
        st.error(err)
    else:
        scaler, kmeans, features = artifact["scaler"], artifact["kmeans"], artifact["features"]

        st.subheader("Which segment does this store belong to?")
        with st.form("clustering_form"):
            c1, c2 = st.columns(2)
            monthly_revenue_kes = c1.number_input("Monthly revenue (KES)", min_value=0.0, value=4000000.0, step=50000.0)
            daily_foot_traffic = c2.number_input("Daily foot traffic", min_value=0.0, value=850.0)
            staff_count = c1.number_input("Staff count", min_value=0.0, value=20.0)
            avg_basket_size_kes = c2.number_input("Average basket size (KES)", min_value=0.0, value=180.0)
            go = st.form_submit_button("Predict segment")

        if go:
            row = pd.DataFrame([{
                "monthly_revenue_kes": monthly_revenue_kes, "daily_foot_traffic": daily_foot_traffic,
                "staff_count": staff_count, "avg_basket_size_kes": avg_basket_size_kes,
            }])[features]
            # Same fitted scaler from training first, then K-Means
            cluster = int(kmeans.predict(scaler.transform(row))[0])
            st.success(f"**Cluster {cluster} -- {CLUSTER_NAMES[cluster]}**")

# -------------------------------------------------------------------------
# 4. RECOMMENDER -- music tracks. Expects the saved
# file to hold the final rules table and a track ID -> name lookup.
# -------------------------------------------------------------------------
def _split_recommender_artifact(artifact):
    """Find the rules table and the ID->name lookup inside the saved dict."""
    rules = artifact.get("rules_final")
    if rules is None:  # fall back to the first table with antecedents/consequents
        rules = next(v for v in artifact.values()
                     if isinstance(v, pd.DataFrame) and "antecedents" in v.columns)
    lookup = next((v for k, v in artifact.items()
                   if "lookup" in k.lower() or "name" in k.lower()), {})
    if isinstance(lookup, pd.DataFrame):
        lookup = lookup.iloc[:, 0]
    return rules, dict(lookup)


with tab_rec:
    artifact, err = load_artifact(RECOMMENDER_FILE)
    if err:
        st.error(err)
    else:
        rules, id_to_name = _split_recommender_artifact(artifact)
        st.subheader("Recommend tracks from what the listener already played")
        with st.form("recommender_form"):
            known = sorted(id_to_name) if id_to_name else []
            picked = st.multiselect("Tracks played", known,
                                    format_func=lambda t: f"{id_to_name.get(t, t)} ({t})")
            top_n = st.number_input("How many recommendations", min_value=1, max_value=20, value=5)
            go = st.form_submit_button("Recommend")

        if go:
            if not picked:
                st.warning("Pick at least one track first.")
            else:
                played = set(picked)
                best = {}
                for _, r in rules.iterrows():
                    if set(r["antecedents"]).issubset(played):
                        for track in r["consequents"]:
                            if track not in played:
                                score = (float(r["confidence"]), float(r["lift"]), float(r["support"]))
                                best[track] = max(best.get(track, score), score)
                ranked = sorted(best.items(), key=lambda kv: tuple(-v for v in kv[1]) + (kv[0],))[:int(top_n)]
                if not ranked:
                    st.info("No rule matches these tracks. Try tracks that appear often together.")
                for track, (conf, lift, support) in ranked:
                    st.write(f"**{id_to_name.get(track, track)}** ({track}) -- "
                             f"confidence {conf:.1%}, lift {lift:.2f}, support {support:.1%}")
