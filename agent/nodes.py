import json
import os
import re
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.state import AgentState
from agent.rag import RegulationRetriever
from preprocessing import preprocess_features

_retriever_instance = None


def _get_retriever() -> RegulationRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RegulationRetriever()
    return _retriever_instance


def _get_api_key() -> str:
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


def parse_borrower_node(state: AgentState) -> AgentState:
    profile = state.get("borrower_profile", {})
    required = [
        "age", "income", "loan_amount", "loan_intent", "credit_score",
        "employment_years", "home_ownership", "interest_rate",
        "credit_history_years", "previous_defaults", "education", "gender"
    ]
    for field in required:
        if field not in profile:
            return {**state, "error": f"Missing required field: {field}"}
    return {**state, "error": None}


def ml_scoring_node(state: AgentState, lr_result, dt_result, xgb_result) -> AgentState:
    if state.get("error"):
        return state

    profile = state["borrower_profile"]
    lr_model, lr_scaler, lr_features = lr_result[0], lr_result[1], lr_result[2]
    dt_model, dt_features = dt_result[0], dt_result[1]
    xgb_model, xgb_features = xgb_result[0], xgb_result[1]

    gender_val = 1 if str(profile.get("gender", "")).lower() == "male" else 0
    default_val = 1 if str(profile.get("previous_defaults", "")).lower() == "yes" else 0
    loan_percent = profile["loan_amount"] / max(1, profile["income"])

    input_df = pd.DataFrame({
        "person_age": [profile["age"]],
        "person_gender": [gender_val],
        "person_education": [profile["education"]],
        "person_income": [profile["income"]],
        "person_emp_exp": [profile["employment_years"]],
        "person_home_ownership": [profile["home_ownership"]],
        "loan_amnt": [profile["loan_amount"]],
        "loan_intent": [profile["loan_intent"]],
        "loan_int_rate": [profile["interest_rate"]],
        "loan_percent_income": [loan_percent],
        "cb_person_cred_hist_length": [profile["credit_history_years"]],
        "credit_score": [profile["credit_score"]],
        "previous_loan_defaults_on_file": [default_val]
    })

    input_processed = preprocess_features(input_df)
    scores = {}

    for model, scaler, features, name in [
        (lr_model, lr_scaler, lr_features, "Logistic Regression"),
        (dt_model, None, dt_features, "Decision Tree"),
        (xgb_model, None, xgb_features, "XGBoost"),
    ]:
        inp = input_processed.copy()
        for col in features:
            if col not in inp.columns:
                inp[col] = 0
        inp = inp[features]
        if scaler is not None:
            inp = scaler.transform(inp)
        prob = model.predict_proba(inp)[0][1]
        scores[name] = round(prob * 100, 2)

    xgb_inp = input_processed.copy()
    for col in xgb_features:
        if col not in xgb_inp.columns:
            xgb_inp[col] = 0
    xgb_inp = xgb_inp[xgb_features]

    importances = xgb_model.feature_importances_
    feat_imp = sorted(zip(xgb_features, importances), key=lambda x: x[1], reverse=True)
    risk_drivers = [
        f.replace("person_", "").replace("loan_", "").replace("_", " ").title()
        for f, _ in feat_imp[:5]
    ]

    xgb_prob = scores["XGBoost"] / 100
    if xgb_prob >= 0.60:
        risk_class = "High Risk"
    elif xgb_prob >= 0.40:
        risk_class = "Medium Risk"
    else:
        risk_class = "Low Risk"

    ml_scores = {
        "model_probabilities": scores,
        "consensus_default_probability": scores["XGBoost"],
        "risk_class": risk_class,
    }
    return {**state, "ml_scores": ml_scores, "risk_drivers": risk_drivers}


def rag_retrieval_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    profile = state["borrower_profile"]
    ml_scores = state["ml_scores"]

    query = (
        f"credit risk {ml_scores['risk_class']} borrower loan amount "
        f"{profile['loan_amount']} income {profile['income']} "
        f"credit score {profile['credit_score']} lending guidelines "
        f"default probability {ml_scores['consensus_default_probability']}%"
    )

    retriever = _get_retriever()
    results = retriever.retrieve(query, k=5)
    return {**state, "rag_context": results}


def llm_assessment_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    from groq import Groq

    profile = state["borrower_profile"]
    ml_scores = state["ml_scores"]
    risk_drivers = state["risk_drivers"]
    rag_context = state["rag_context"]

    reg_text = "\n\n".join([
        f"[Source: {r['source']}]\n{r['text']}" for r in rag_context
    ])

    system_prompt = (
        "You are a credit risk assessment AI assistant for a regulated financial institution. "
        "Generate structured lending assessment reports based on ML model outputs and regulatory guidelines.\n\n"
        "MANDATORY RULES:\n"
        "1. Base ALL risk decisions ONLY on objective financial metrics: credit score, income, "
        "loan amount, employment history, debt-to-income ratio, and payment history.\n"
        "2. NEVER consider or mention gender, religion, caste, ethnicity, or any protected attribute.\n"
        "3. Cite ONLY sources explicitly given in the regulatory context below.\n"
        "4. When data is ambiguous or contradictory, explicitly state the uncertainty.\n"
        "5. Always include the legal disclaimer.\n"
        "6. Return ONLY valid JSON with no markdown fences, no extra text before or after.\n\n"
        'Return this exact JSON structure:\n'
        '{\n'
        '  "borrower_summary": "2-3 sentence summary of the borrower financial profile",\n'
        '  "risk_analysis": "3-4 sentence analysis of key risk drivers and their implications",\n'
        '  "lending_decision": "APPROVE or CONDITIONAL APPROVE or DECLINE",\n'
        '  "decision_rationale": "2-3 sentence justification based solely on financial data",\n'
        '  "conditions": ["list conditions if CONDITIONAL APPROVE, else empty list"],\n'
        '  "regulatory_references": ["2-3 specific citations from provided regulatory context"],\n'
        '  "responsible_ai_note": "1 sentence on fair lending adherence",\n'
        '  "disclaimer": "Legal disclaimer for AI-generated credit assessments"\n'
        '}'
    )

    user_message = (
        f"BORROWER FINANCIAL PROFILE:\n"
        f"- Annual Income: ${profile['income']:,}\n"
        f"- Loan Amount Requested: ${profile['loan_amount']:,}\n"
        f"- Loan-to-Income Ratio: {profile['loan_amount'] / max(1, profile['income']):.2%}\n"
        f"- Credit Score: {profile['credit_score']}\n"
        f"- Employment Experience: {profile['employment_years']} years\n"
        f"- Home Ownership: {profile['home_ownership']}\n"
        f"- Loan Purpose: {profile['loan_intent']}\n"
        f"- Interest Rate: {profile['interest_rate']}%\n"
        f"- Credit History Length: {profile['credit_history_years']} years\n"
        f"- Previous Loan Defaults: {profile['previous_defaults']}\n"
        f"- Education Level: {profile['education']}\n\n"
        f"ML MODEL ASSESSMENT:\n"
        f"- XGBoost Default Probability: {ml_scores['consensus_default_probability']}%\n"
        f"- Logistic Regression Probability: {ml_scores['model_probabilities']['Logistic Regression']}%\n"
        f"- Decision Tree Probability: {ml_scores['model_probabilities']['Decision Tree']}%\n"
        f"- Consensus Risk Classification: {ml_scores['risk_class']}\n"
        f"- Top Risk Drivers: {', '.join(risk_drivers)}\n\n"
        f"REGULATORY CONTEXT:\n{reg_text}\n\n"
        "Generate the structured credit assessment report as JSON."
    )

    api_key = _get_api_key()
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=1500,
    )

    raw_output = response.choices[0].message.content.strip()

    try:
        report = json.loads(raw_output)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if match:
            try:
                report = json.loads(match.group())
            except json.JSONDecodeError:
                report = {"error": "Failed to parse LLM response", "raw_output": raw_output}
        else:
            report = {"error": "Failed to parse LLM response", "raw_output": raw_output}

    return {**state, "assessment_report": report}


def format_report_node(state: AgentState) -> AgentState:
    return state
