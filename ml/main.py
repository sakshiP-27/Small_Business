import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

print("LOAN ACCEPTANCE PREDICTION MODEL")

# ── 1. LOAD DATA ─────────────────────────────────────────────
print("\n[1/6] Loading data...")
accepted_df = pd.read_csv("sme1.csv")
rejected_df = pd.read_csv("sme2.csv")
print(f"  Accepted: {len(accepted_df):,} | Rejected: {len(rejected_df):,}")

SAMPLE_SIZE = 500_000
accepted_df = accepted_df.sample(n=min(SAMPLE_SIZE, len(accepted_df)), random_state=42)
rejected_df = rejected_df.sample(n=min(SAMPLE_SIZE, len(rejected_df)), random_state=42)
print(f"  Sampled  → Accepted: {len(accepted_df):,} | Rejected: {len(rejected_df):,}")

# ── 2. MAP TO SHARED FEATURE SPACE ───────────────────────────
# sme2 only has: Amount Requested, Debt-To-Income Ratio,
#                Employment Length, Risk_Score
# We ONLY train on features genuinely present in both datasets.
# Everything else is used at inference time but not for the classifier.
print("\n[2/6] Mapping to shared feature space (real features only)...")

EMP_MAPPING = {
    '< 1 year': 1, '1 year': 2, '2 years': 3, '3 years': 4,
    '4 years': 5, '5 years': 6, '6 years': 7, '7 years': 8,
    '8 years': 9, '9 years': 10, '10+ years': 11,
    'Unknown': 0, 'nan': 0, 'n/a': 0
}

def map_accepted(df):
    out = pd.DataFrame()
    out['loan_amnt']      = pd.to_numeric(df['loan_amnt'], errors='coerce')
    out['dti']            = pd.to_numeric(df['dti'], errors='coerce')
    out['fico_range_low'] = pd.to_numeric(df['fico_range_low'], errors='coerce')
    out['fico_range_high']= pd.to_numeric(df['fico_range_high'], errors='coerce')
    out['emp_stability']  = df['emp_length'].astype(str).str.strip().map(EMP_MAPPING).fillna(0)
    out['loan_decision']  = 'Accepted'
    return out

def map_rejected(df):
    out = pd.DataFrame()
    out['loan_amnt']      = pd.to_numeric(df['Amount Requested'], errors='coerce')
    out['dti']            = (
        df['Debt-To-Income Ratio'].astype(str).str.rstrip('%')
        .pipe(pd.to_numeric, errors='coerce')
    )
    out['fico_range_low'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
    out['fico_range_high']= pd.to_numeric(df['Risk_Score'], errors='coerce')
    out['emp_stability']  = df['Employment Length'].astype(str).str.strip().map(EMP_MAPPING).fillna(0)
    out['loan_decision']  = 'Rejected'
    return out

accepted_mapped = map_accepted(accepted_df)
rejected_mapped = map_rejected(rejected_df)

df = pd.concat([accepted_mapped, rejected_mapped], ignore_index=True)
print(f"  Combined: {len(df):,} | Accepted: {(df.loan_decision=='Accepted').sum():,} | Rejected: {(df.loan_decision=='Rejected').sum():,}")

# ── 3. CLEAN ─────────────────────────────────────────────────
print("\n[3/6] Cleaning...")
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].median())
print(f"  Nulls remaining: {df.isnull().sum().sum()}")

# ── 4. FEATURE ENGINEERING ───────────────────────────────────
print("\n[4/6] Engineering features...")
# fico midpoint
df['fico_mid'] = (df['fico_range_low'] + df['fico_range_high']) / 2
# dti buckets
df['dti_bucket'] = pd.cut(df['dti'].clip(0, 100),
    bins=[0, 10, 20, 30, 40, 100],
    labels=['0-10', '10-20', '20-30', '30-40', '40+'])
df['dti_bucket'] = df['dti_bucket'].astype(str)

# ── 5. TRAIN ─────────────────────────────────────────────────
print("\n[5/6] Training model...")

FEATURES = ['loan_amnt', 'dti', 'fico_range_low', 'fico_range_high',
            'fico_mid', 'emp_stability', 'dti_bucket']

X = df[FEATURES]
y = df['loan_decision']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
print(f"  Classes: {le.classes_}  (0={le.classes_[0]}, 1={le.classes_[1]})")

cat_cols = ['dti_bucket']
X_train_p = pd.get_dummies(X_train.copy(), columns=cat_cols, drop_first=True)
X_test_p  = pd.get_dummies(X_test.copy(),  columns=cat_cols, drop_first=True)
X_train_p, X_test_p = X_train_p.align(X_test_p, join='left', axis=1, fill_value=0)

for frame in [X_train_p, X_test_p]:
    frame.columns = (frame.columns
        .str.replace('[','_',regex=False).str.replace(']','_',regex=False)
        .str.replace('<','_',regex=False).str.replace('>','_',regex=False))

neg = (y_train_enc == 0).sum()
pos = (y_train_enc == 1).sum()
spw = neg / pos if pos > 0 else 1

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    max_depth=4,
    learning_rate=0.05,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_p, y_train_enc,
          eval_set=[(X_test_p, y_test_enc)],
          verbose=False)

y_pred      = model.predict(X_test_p)
y_pred_prob = model.predict_proba(X_test_p)[:, 1]
acc = accuracy_score(y_test_enc, y_pred)
roc = roc_auc_score(y_test_enc, y_pred_prob)
print(f"  Accuracy: {acc*100:.2f}%  |  ROC-AUC: {roc:.4f}")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))
print(f"  Prob dist → mean: {y_pred_prob.mean():.3f}  std: {y_pred_prob.std():.3f}  min: {y_pred_prob.min():.3f}  max: {y_pred_prob.max():.3f}")

# ── 6. SHAP + SAVE ───────────────────────────────────────────
print("\n[6/6] SHAP explainer + saving...")

sample_size = min(500, len(X_train_p))
X_shap = X_train_p.sample(n=sample_size, random_state=42)
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_shap)

feature_importance = pd.DataFrame({
    'feature':    X_shap.columns,
    'importance': np.abs(shap_vals).mean(axis=0)
}).sort_values('importance', ascending=False)

print("\nTop features:")
print(feature_importance.to_string(index=False))

# ── Build full feature_columns list for inference ────────────
# At inference time the backend sends all 25 fields.
# We need to store the TRAINING columns so preprocess() can align.
# Also store the full accepted feature columns for SHAP explanations.
full_accepted_cols = [
    'loan_amnt','dti','fico_range_low','fico_range_high','fico_mid',
    'emp_stability','dti_bucket_10-20','dti_bucket_20-30',
    'dti_bucket_30-40','dti_bucket_40_'
]

pkg = {
    'model':              model,
    'label_encoder':      le,
    'feature_columns':    X_train_p.columns.tolist(),
    'categorical_columns': cat_cols,
    'explainer':          explainer,
    'feature_importance': feature_importance,
    'emp_mapping':        EMP_MAPPING,
}
joblib.dump(pkg, 'loan_model.pkl')
joblib.dump(le,  'label_encoder.pkl')

print("\n  ✓ loan_model.pkl saved")
print(f"\nDONE → Accuracy: {acc*100:.2f}%  ROC-AUC: {roc:.4f}")
