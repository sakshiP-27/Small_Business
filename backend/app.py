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
    pos = sv < 0  # negative SHAP = reduces rejection = good
    good, bad = "✓", "✗"
    mark = good if pos else bad

    try: val_f = float(val)
    except: val_f = 0

    templates = {
        'fico_range_high':  f"{mark} FICO score (high) of {int(val_f)}",
        'fico_range_low':   f"{mark} FICO score (low) of {int(val_f)}",
        'fico_mid':         f"{mark} FICO midpoint of {int(val_f)}",
        'emp_stability':    f"{mark} Employment stability score of {int(val_f)}",
        'dti_bucket':       f"{mark} DTI in range {feat.replace('dti_bucket_','').replace('_','+').replace('-',' – ')}%",
        'dti':              f"{mark} Debt-to-income ratio of {val_f:.1f}%",
        'loan_amnt':        f"{mark} Loan amount of ${val_f:,.0f}",
    }
    fn = feat.lower()
    # match dti_bucket before dti
    if 'dti_bucket' in fn:
        label = feat.replace('dti_bucket_','').replace('_','+')
        return f"{mark} DTI in range {label}%"
    for key, text in templates.items():
        if fn == key or fn.startswith(key):
            return text
    return f"{mark} {feat.replace('_',' ').title()} = {val_f:.2f}"

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

    # Coerce all numeric fields — empty string → 0
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
    # (fields not in the model but useful to show)
    extra_positive, extra_negative = [], []

    annual_inc = body.get('annual_inc', 0)
    if annual_inc >= 60000:
        extra_positive.append(f"✓ Annual income of ${annual_inc:,.0f}")
    elif annual_inc > 0:
        extra_negative.append(f"✗ Annual income of ${annual_inc:,.0f}")

    revol_util = body.get('revol_util', 0)
    if revol_util <= 30:
        extra_positive.append(f"✓ Revolving utilization of {revol_util:.1f}%")
    elif revol_util >= 70:
        extra_negative.append(f"✗ High revolving utilization of {revol_util:.1f}%")

    inq = body.get('inq_last_6mths', 0)
    if inq >= 3:
        extra_negative.append(f"✗ {int(inq)} credit inquiries in last 6 months")
    elif inq == 0:
        extra_positive.append("✓ No recent credit inquiries")

    delinq = body.get('delinq_2yrs', 0)
    if delinq > 0:
        extra_negative.append(f"✗ {int(delinq)} delinquencies in last 2 years")

    pub_rec = body.get('pub_rec', 0)
    if pub_rec > 0:
        extra_negative.append(f"✗ {int(pub_rec)} public record(s) on file")

    revol_bal = body.get('revol_bal', 0)
    if revol_bal > 0:
        extra_positive.append(f"✓ Revolving balance of ${revol_bal:,.0f}")

    installment = body.get('installment', 0)
    if installment > 0:
        extra_positive.append(f"✓ Monthly installment of ${installment:,.0f}")

    int_rate = body.get('int_rate', 0)
    if int_rate > 0:
        if int_rate <= 10:
            extra_positive.append(f"✓ Low interest rate of {int_rate:.2f}%")
        elif int_rate >= 20:
            extra_negative.append(f"✗ High interest rate of {int_rate:.2f}%")

    open_acc = body.get('open_acc', 0)
    total_acc = body.get('total_acc', 0)
    if open_acc > 0:
        extra_positive.append(f"✓ {int(open_acc)} open credit accounts")
    if total_acc > 0:
        extra_positive.append(f"✓ {int(total_acc)} total credit accounts")

    # Merge model SHAP explanations with extra context, deduplicate
    seen = set()
    def dedup(items):
        out = []
        for item in items:
            key = item[2:]  # strip ✓/✗ prefix for dedup key
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
