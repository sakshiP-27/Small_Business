from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# ── Load model ───────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml', 'loan_model.pkl')
pkg = joblib.load(MODEL_PATH)

model           = pkg['model']
le              = pkg['label_encoder']
feature_columns = pkg['feature_columns']
explainer       = pkg['explainer']
EMP_MAPPING     = pkg.get('emp_mapping', {
    '< 1 year': 1, '1 year': 2, '2 years': 3, '3 years': 4,
    '4 years': 5, '5 years': 6, '6 years': 7, '7 years': 8,
    '8 years': 9, '9 years': 10, '10+ years': 11, 'Unknown': 0
})

REJECTED_IDX = list(le.classes_).index('Rejected')

# ── Helpers ──────────────────────────────────────────────────
def credit_score_from_prob(rejection_prob: float) -> int:
    score = 850 - (rejection_prob * 550)
    return int(np.clip(score, 300, 850))

def risk_band(score: int) -> str:
    if score >= 750: return 'Low Risk'
    if score >= 650: return 'Medium Risk'
    return 'High Risk'

def plain_english(feat: str, val, sv: float) -> str:
    pos = sv < 0  # negative SHAP = reduces rejection probability = good
    try: val_f = float(val)
    except: val_f = 0
    fn = feat.lower()

    if 'dti_bucket' in fn:
        label = feat.replace('dti_bucket_', '').replace('_', '+')
        if pos:
            return f"Your debt-to-income ratio falls in the {label}% range, which is within an acceptable range for most lenders."
        else:
            return f"Your debt-to-income ratio falls in the {label}% range. A high DTI means a large portion of your income is already committed to debt, which concerns lenders."

    if fn in ('fico_range_low', 'fico_range_high', 'fico_mid'):
        score = int(val_f)
        if pos:
            if score >= 750:
                return f"Your FICO score of {score} is excellent. Scores above 750 place you in the top tier of borrowers and strongly support loan approval."
            elif score >= 670:
                return f"Your FICO score of {score} is good. This is above the typical approval threshold and reflects a solid credit history."
            else:
                return f"Your FICO score of {score} is fair. While not disqualifying, improving it above 670 would significantly strengthen your application."
        else:
            if score < 580:
                return f"Your FICO score of {score} is poor. This is below most lenders' minimum thresholds and is a primary reason for concern in your application."
            elif score < 670:
                return f"Your FICO score of {score} is below average. Lenders typically prefer scores above 670 — this is likely one of the biggest factors working against your application."
            else:
                return f"Your FICO score of {score} is contributing negatively relative to other applicants in the model's assessment."

    if fn == 'emp_stability':
        if pos:
            if val_f >= 11:
                return "Your employment history of 10+ years is exceptional. Long-term stable employment is one of the strongest signals of financial reliability."
            elif val_f >= 6:
                return f"Your employment stability of around {int(val_f) - 1}–{int(val_f)} years is solid. Consistent employment history reassures lenders of steady income."
            else:
                return "Your employment history shows some stability, which is a positive factor in your assessment."
        else:
            if val_f <= 1:
                return "Your employment history is less than 1 year. Lenders prefer at least 2 years of stable employment — short tenure raises questions about income consistency."
            else:
                return f"Your employment stability of around {int(val_f) - 1}–{int(val_f)} years is below what lenders typically prefer. Longer tenure strengthens your application."

    if fn == 'dti':
        if pos:
            if val_f <= 15:
                return f"Your debt-to-income ratio of {val_f:.1f}% is very low. Only a small portion of your income goes toward debt, leaving plenty of room for a new loan."
            else:
                return f"Your debt-to-income ratio of {val_f:.1f}% is within an acceptable range for most lenders."
        else:
            if val_f >= 40:
                return f"Your debt-to-income ratio of {val_f:.1f}% is very high. More than 40% of your income is already going toward debt payments, which makes lenders hesitant to add more."
            else:
                return f"Your debt-to-income ratio of {val_f:.1f}% is elevated. Lenders generally prefer a DTI below 36% — reducing existing debt would improve your standing."

    if fn == 'loan_amnt':
        if pos:
            return f"The requested loan amount of ${val_f:,.0f} is reasonable relative to your financial profile and doesn't appear to be a strain on your capacity."
        else:
            return f"The requested loan amount of ${val_f:,.0f} is high relative to your current financial profile. Requesting a smaller amount may improve your approval chances."

    # fallback
    label = feat.replace('_', ' ').title()
    if pos:
        return f"Your {label} is a positive factor in your credit assessment."
    else:
        return f"Your {label} is flagged as a concern in your credit assessment."

def preprocess(body: dict) -> pd.DataFrame:
    fico_low  = float(body.get('fico_range_low', 0) or 0)
    fico_high = float(body.get('fico_range_high', 0) or 0)
    dti       = float(body.get('dti', 0) or 0)
    loan_amnt = float(body.get('loan_amnt', 0) or 0)
    emp_str   = str(body.get('emp_length', 'Unknown')).strip()
    emp_stab  = EMP_MAPPING.get(emp_str, 0)

    fico_mid  = (fico_low + fico_high) / 2

    dti_clipped = min(max(dti, 0), 100)
    if dti_clipped <= 10:   dti_bucket = '0-10'
    elif dti_clipped <= 20: dti_bucket = '10-20'
    elif dti_clipped <= 30: dti_bucket = '20-30'
    elif dti_clipped <= 40: dti_bucket = '30-40'
    else:                   dti_bucket = '40+'

    row = {
        'loan_amnt':      loan_amnt,
        'dti':            dti,
        'fico_range_low': fico_low,
        'fico_range_high':fico_high,
        'fico_mid':       fico_mid,
        'emp_stability':  emp_stab,
        f'dti_bucket_{dti_bucket}': 1,
    }

    df = pd.DataFrame([row])
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

# ── Routes ───────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/predict', methods=['POST'])
def predict():
    body = request.get_json(force=True)
    if not body:
        return jsonify({"error": "Request body is required"}), 400

    # Coerce all numeric fields → empty string → 0
    NUMERIC = ['loan_amnt','int_rate','installment','annual_inc','dti',
               'inq_last_6mths','delinq_2yrs','open_acc','total_acc',
               'revol_bal','revol_util','pub_rec','pub_rec_bankruptcies',
               'fico_range_low','fico_range_high','delinq_amnt',
               'num_tl_30dpd','num_tl_90g_dpd_24m']
    for f in NUMERIC:
        v = body.get(f, '')
        body[f] = float(v) if v not in ('', None) else 0

    try:
        X = preprocess(body)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 422

    pred_enc   = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    loan_status = le.inverse_transform([pred_enc])[0]

    rejection_prob      = float(pred_proba[REJECTED_IDX])
    default_probability = round(rejection_prob, 4)
    score               = credit_score_from_prob(rejection_prob)
    band                = risk_band(score)

    # SHAP
    shap_vals    = explainer.shap_values(X)[0]
    feature_vals = X.iloc[0]

    pairs = sorted(
        zip(feature_columns, feature_vals, shap_vals),
        key=lambda x: abs(x[2]), reverse=True
    )

    positive, negative = [], []
    for feat, val, sv in pairs[:10]:
        if abs(sv) < 1e-4: continue
        text = plain_english(feat, val, sv)
        if sv < 0:
            positive.append(text)
        else:
            negative.append(text)

    # Also add human-readable context from the full payload
    extra_positive, extra_negative = [], []

    annual_inc = body.get('annual_inc', 0)
    if annual_inc >= 100000:
        extra_positive.append(f"Your annual income of ${annual_inc:,.0f} is strong and well above typical lender thresholds, which significantly improves your borrowing capacity.")
    elif annual_inc >= 60000:
        extra_positive.append(f"Your annual income of ${annual_inc:,.0f} meets most lenders' minimum requirements and supports your requested loan amount.")
    elif annual_inc > 0:
        extra_negative.append(f"Your annual income of ${annual_inc:,.0f} may be considered low relative to the loan amount requested. Lenders typically prefer higher income to ensure repayment ability.")

    revol_util = body.get('revol_util', 0)
    if revol_util <= 10:
        extra_positive.append(f"Your credit utilization is just {revol_util:.1f}% — excellent. Keeping utilization below 10% signals to lenders that you're not over-relying on credit.")
    elif revol_util <= 30:
        extra_positive.append(f"Your credit utilization of {revol_util:.1f}% is healthy. Staying under 30% is a positive signal that you manage your credit responsibly.")
    elif revol_util <= 60:
        extra_negative.append(f"Your credit utilization is {revol_util:.1f}%, which is moderate but elevated. Lenders prefer this below 30% — consider paying down balances to improve your profile.")
    else:
        extra_negative.append(f"Your credit utilization of {revol_util:.1f}% is high. This suggests you're heavily reliant on available credit, which is a red flag for lenders. Paying down revolving balances would meaningfully improve your score.")

    inq = body.get('inq_last_6mths', 0)
    if inq == 0:
        extra_positive.append("You have no recent credit inquiries in the last 6 months. This tells lenders you haven't been aggressively seeking new credit, which is a positive sign.")
    elif inq <= 2:
        extra_negative.append(f"You have {int(inq)} credit inquiry(s) in the last 6 months. Each inquiry can slightly lower your score — multiple inquiries suggest you may be seeking credit urgently.")
    else:
        extra_negative.append(f"You have {int(inq)} credit inquiries in the last 6 months. This is a significant concern — it signals financial stress or urgency to lenders and can noticeably hurt your application.")

    delinq = body.get('delinq_2yrs', 0)
    if delinq == 0:
        extra_positive.append("You have no delinquencies in the past 2 years. A clean payment history is one of the most important factors lenders look at.")
    elif delinq == 1:
        extra_negative.append(f"You have 1 delinquency in the last 2 years. Even a single missed payment can raise concerns — lenders want to see a consistent on-time payment history.")
    else:
        extra_negative.append(f"You have {int(delinq)} delinquencies in the last 2 years. This is a serious negative signal. It indicates a pattern of missed payments, which significantly increases perceived risk.")

    pub_rec = body.get('pub_rec', 0)
    if pub_rec == 0:
        extra_positive.append("Your public record is clean — no bankruptcies, liens, or judgments on file. This is an important trust signal for lenders.")
    else:
        extra_negative.append(f"You have {int(pub_rec)} public record(s) on file (such as bankruptcies or court judgments). These are serious derogatory marks that can heavily impact approval chances.")

    int_rate = body.get('int_rate', 0)
    if int_rate > 0:
        if int_rate <= 8:
            extra_positive.append(f"Your current interest rate of {int_rate:.1f}% is very low, reflecting a strong credit history and low risk profile.")
        elif int_rate <= 15:
            extra_positive.append(f"Your interest rate of {int_rate:.1f}% is within a normal range, suggesting a reasonably healthy credit profile.")
        elif int_rate <= 20:
            extra_negative.append(f"Your interest rate of {int_rate:.1f}% is above average. This may reflect past credit issues and can signal elevated risk to lenders.")
        else:
            extra_negative.append(f"Your interest rate of {int_rate:.1f}% is high, which typically indicates a riskier credit profile. Lenders may view this as a sign of prior financial difficulty.")

    open_acc = body.get('open_acc', 0)
    total_acc = body.get('total_acc', 0)
    if total_acc >= 10 and open_acc >= 3:
        extra_positive.append(f"You have {int(open_acc)} open accounts out of {int(total_acc)} total. A diverse, established credit history like this demonstrates experience managing multiple credit lines.")
    elif total_acc < 3:
        extra_negative.append(f"You only have {int(total_acc)} total credit account(s). A thin credit history makes it harder for lenders to assess your reliability — building more credit history over time will help.")

    revol_bal = body.get('revol_bal', 0)
    if revol_bal > 50000:
        extra_negative.append(f"Your revolving balance of ${revol_bal:,.0f} is quite high. Large outstanding balances increase your debt load and may concern lenders about your ability to take on more.")
    elif revol_bal > 0:
        extra_positive.append(f"Your revolving balance of ${revol_bal:,.0f} is manageable and within a reasonable range.")

    # Merge model SHAP explanations with extra context, deduplicate
    seen = set()
    def dedup(items):
        out = []
        for item in items:
            key = item.strip()
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out

    final_positive = dedup(positive + extra_positive)
    final_negative = dedup(negative + extra_negative)

    return jsonify({
        "credit_score":        score,
        "default_probability": default_probability,
        "risk_band":           band,
        "loan_status":         loan_status,
        "explanations": {
            "positive": final_positive[:10],
            "negative": final_negative[:10],
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=8000)
