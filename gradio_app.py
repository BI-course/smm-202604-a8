
import gradio as gr
import requests

API_URL = "http://127.0.0.1:5000/api/v1/models/sme-credit-risk-classifier/predictions"

def predict_risk(
    sector,
    county,
    quarter,
    loan_purpose,
    owner_education_level,
    business_age_years,
    num_employees,
    monthly_revenue_kes,
    loan_amount_kes,
    loan_term_months,
    interest_rate_pct,
    collateral_value_kes,
    debt_to_income_ratio,
    credit_score,
    num_previous_loans,
    missed_payments_count,
    online_presence_score,
    customer_satisfaction_score,
    competitor_density
):
    data = {
        "sector": sector,
        "county": county,
        "quarter": quarter,
        "loan_purpose": loan_purpose,
        "owner_education_level": owner_education_level,
        "business_age_years": business_age_years,
        "num_employees": num_employees,
        "monthly_revenue_kes": monthly_revenue_kes,
        "loan_amount_kes": loan_amount_kes,
        "loan_term_months": loan_term_months,
        "interest_rate_pct": interest_rate_pct,
        "collateral_value_kes": collateral_value_kes,
        "debt_to_income_ratio": debt_to_income_ratio,
        "credit_score": credit_score,
        "num_previous_loans": num_previous_loans,
        "missed_payments_count": missed_payments_count,
        "online_presence_score": online_presence_score,
        "customer_satisfaction_score": customer_satisfaction_score,
        "competitor_density": competitor_density
    }

    try:
        response = requests.post(API_URL, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()

        prediction = result.get("prediction", "Unknown")
        probabilities = result.get("probabilities", {})

        formatted_probabilities = "\n".join(
            f"{category}: {probability:.2%}"
            for category, probability in probabilities.items()
        )

        return prediction, formatted_probabilities, result

    except requests.exceptions.ConnectionError:
        return (
            "API unavailable",
            "Could not connect to the Flask server.",
            {"error": "Start api.py first."}
        )

    except requests.exceptions.RequestException as e:
        return (
            "Request failed",
            str(e),
            {"error": str(e)}
        )


with gr.Blocks(title="SME Credit Risk Classifier") as demo:

    gr.Markdown("# SME Credit Risk Classification")
    gr.Markdown(
        "Enter business and loan information to obtain "
        "a prediction from the trained classification model."
    )

    with gr.Row():
        sector = gr.Textbox(label="Business Sector", value="Retail")
        county = gr.Textbox(label="County", value="Nairobi")
        quarter = gr.Textbox(label="Quarter", value="Q4")

    with gr.Row():
        loan_purpose = gr.Textbox(
            label="Loan Purpose", value="Expansion"
        )
        owner_education_level = gr.Dropdown(
            label="Owner Education Level",
            choices=[
                "Primary",
                "Secondary",
                "Certificate",
                "Diploma",
                "Bachelors",
                "Masters",
                "Doctorate"
            ],
            value="Diploma"
        )

    gr.Markdown("### Business Information")

    with gr.Row():
        business_age_years = gr.Number(
            label="Business Age (Years)", value=5
        )
        num_employees = gr.Number(
            label="Number of Employees", value=10
        )
        monthly_revenue_kes = gr.Number(
            label="Monthly Revenue (KES)", value=250000
        )

    gr.Markdown("### Loan Information")

    with gr.Row():
        loan_amount_kes = gr.Number(
            label="Loan Amount (KES)", value=500000
        )
        loan_term_months = gr.Number(
            label="Loan Term (Months)", value=12
        )
        interest_rate_pct = gr.Number(
            label="Interest Rate (%)", value=14.5
        )

    with gr.Row():
        collateral_value_kes = gr.Number(
            label="Collateral Value (KES)", value=600000
        )
        debt_to_income_ratio = gr.Number(
            label="Debt-to-Income Ratio", value=0.35
        )
        credit_score = gr.Number(
            label="Credit Score", value=680
        )

    gr.Markdown("### Other Indicators")

    with gr.Row():
        num_previous_loans = gr.Number(
            label="Previous Loans", value=2
        )
        missed_payments_count = gr.Number(
            label="Missed Payments", value=0
        )
        online_presence_score = gr.Number(
            label="Online Presence Score", value=7
        )

    with gr.Row():
        customer_satisfaction_score = gr.Number(
            label="Customer Satisfaction Score", value=8
        )
        competitor_density = gr.Number(
            label="Competitor Density", value=5
        )

    predict_button = gr.Button("Predict Credit Risk")

    gr.Markdown("### Prediction Result")

    prediction_output = gr.Textbox(
        label="Predicted Category"
    )

    probability_output = gr.Textbox(
        label="Class Probabilities"
    )

    raw_output = gr.JSON(label="API Response")

    predict_button.click(
        fn=predict_risk,
        inputs=[
            sector,
            county,
            quarter,
            loan_purpose,
            owner_education_level,
            business_age_years,
            num_employees,
            monthly_revenue_kes,
            loan_amount_kes,
            loan_term_months,
            interest_rate_pct,
            collateral_value_kes,
            debt_to_income_ratio,
            credit_score,
            num_previous_loans,
            missed_payments_count,
            online_presence_score,
            customer_satisfaction_score,
            competitor_density
        ],
        outputs=[
            prediction_output,
            probability_output,
            raw_output
        ]
    )

demo.launch()
