# ============================================================
# DIABETES RISK PREDICTION — End-to-End ML Pipeline
# Dataset: Pima Indians Diabetes Database (UCI / Kaggle)
# Author: Chantale Jepleting Ruto
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, roc_curve, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
import shap
import os

SEED = 42
np.random.seed(SEED)
os.makedirs('../outputs', exist_ok=True)

print("=" * 55)
print("  DIABETES RISK PREDICTION PIPELINE")
print("=" * 55)

# -------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------
print("\n[1/8] Loading dataset...")

df = pd.read_csv('../data/diabetes.csv')

# In this dataset, 0 values in clinical features are biologically
# implausible (e.g. Glucose=0, BMI=0) — replace with NaN for imputation
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

print(f"    Shape: {df.shape} | Diabetic: {df['Outcome'].sum()} | Non-diabetic: {(df['Outcome']==0).sum()}")
print(f"    Missing values introduced (implausible zeros):\n    {df[zero_cols].isnull().sum().to_dict()}")

# -------------------------------------------------------
# 2. EDA PLOTS
# -------------------------------------------------------
print("\n[2/8] Generating EDA plots...")

# Class distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['Outcome'].value_counts().plot(kind='bar', ax=axes[0],
    color=['#228B22','#C8102E'], edgecolor='black')
axes[0].set_title('Target Class Distribution')
axes[0].set_xticklabels(['Non-Diabetic (0)', 'Diabetic (1)'], rotation=0)
axes[0].set_ylabel('Count')

# Age distribution by outcome
df[df['Outcome']==1]['Age'].hist(alpha=0.6, bins=20, ax=axes[1], color='#C8102E', label='Diabetic')
df[df['Outcome']==0]['Age'].hist(alpha=0.6, bins=20, ax=axes[1], color='#228B22', label='Non-Diabetic')
axes[1].set_title('Age Distribution by Outcome')
axes[1].set_xlabel('Age'); axes[1].legend()
plt.tight_layout()
plt.savefig('../outputs/01_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
sns.heatmap(numeric_df.corr(), mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig('../outputs/02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature distributions by outcome
features = [c for c in df.columns if c != 'Outcome']
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, feat in enumerate(features):
    ax = axes[i//4][i%4]
    df[df['Outcome']==1][feat].hist(alpha=0.6, bins=25, ax=ax, color='#C8102E', label='Diabetic')
    df[df['Outcome']==0][feat].hist(alpha=0.6, bins=25, ax=ax, color='#228B22', label='Non-Diabetic')
    ax.set_title(feat); ax.legend(fontsize=7)
plt.suptitle('Feature Distributions by Outcome', fontsize=14)
plt.tight_layout()
plt.savefig('../outputs/03_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# Boxplots
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, feat in enumerate(features):
    ax = axes[i//4][i%4]
    df.boxplot(column=feat, by='Outcome', ax=ax,
               boxprops=dict(color='#003087'),
               medianprops=dict(color='#C8102E', linewidth=2))
    ax.set_title(feat); ax.set_xlabel('Outcome (0=No, 1=Yes)')
plt.suptitle('Feature Boxplots by Diabetes Outcome', fontsize=14)
plt.tight_layout()
plt.savefig('../outputs/04_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: 01–04 EDA plots")

# -------------------------------------------------------
# 3. PREPROCESSING
# -------------------------------------------------------
print("\n[3/8] Preprocessing...")

X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"    Missing values before MICE: {X.isnull().sum().sum()}")
mice = IterativeImputer(random_state=SEED, max_iter=10)
X_imputed = pd.DataFrame(mice.fit_transform(X), columns=X.columns)
print(f"    Missing values after MICE:  {X_imputed.isnull().sum().sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=SEED, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"    Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# -------------------------------------------------------
# 4. MODEL TRAINING
# -------------------------------------------------------
print("\n[4/8] Training models...")

def evaluate_model(name, model, Xtr, Xte, y_tr, y_te):
    model.fit(Xtr, y_tr)
    y_prob = model.predict_proba(Xte)[:, 1]
    y_pred = model.predict(Xte)
    auc = roc_auc_score(y_te, y_prob)
    cv_auc = cross_val_score(model, Xtr, y_tr, cv=5, scoring='roc_auc').mean()
    print(f"\n  {name}")
    print(f"    Test AUC: {auc:.4f} | CV AUC (5-fold): {cv_auc:.4f}")
    print(classification_report(y_te, y_pred, target_names=['Non-Diabetic','Diabetic']))
    return model, y_prob, auc, cv_auc

lr = LogisticRegression(random_state=SEED, max_iter=1000, class_weight='balanced')
lr_model, lr_probs, lr_auc, lr_cv = evaluate_model(
    "Logistic Regression", lr, X_train_scaled, X_test_scaled, y_train, y_test)

rf = RandomForestClassifier(n_estimators=200, random_state=SEED, class_weight='balanced')
rf_model, rf_probs, rf_auc, rf_cv = evaluate_model(
    "Random Forest", rf, X_train.values, X_test.values, y_train, y_test)

scale_pos = (y_train==0).sum() / (y_train==1).sum()
xgb = XGBClassifier(n_estimators=200, random_state=SEED,
                     scale_pos_weight=scale_pos, eval_metric='auc', verbosity=0)
xgb_model, xgb_probs, xgb_auc, xgb_cv = evaluate_model(
    "XGBoost", xgb, X_train.values, X_test.values, y_train, y_test)

# -------------------------------------------------------
# 5. MODEL COMPARISON
# -------------------------------------------------------
print("\n[5/8] Generating comparison plots...")

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Test AUC': [lr_auc, rf_auc, xgb_auc],
    'CV AUC': [lr_cv, rf_cv, xgb_cv]
}).sort_values('Test AUC', ascending=False)

# ROC curves
plt.figure(figsize=(8, 6))
for name, probs in [("Logistic Regression", lr_probs),
                     ("Random Forest", rf_probs),
                     ("XGBoost", xgb_probs)]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val = results[results['Model']==name]['Test AUC'].values[0]
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_val:.3f})")
plt.plot([0,1],[0,1],'k--', lw=1)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curves — Model Comparison')
plt.legend(loc='lower right'); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/05_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# Bar chart
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(results)); width = 0.35
bars1 = ax.bar(x - width/2, results['Test AUC'], width, label='Test AUC', color='#C8102E', alpha=0.85)
bars2 = ax.bar(x + width/2, results['CV AUC'], width, label='CV AUC', color='#228B22', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(results['Model']); ax.set_ylim(0.5, 1.0)
ax.axhline(0.8, color='gray', linestyle='--', linewidth=1, label='AUC=0.80 threshold')
ax.set_title('Model Performance Comparison'); ax.legend()
ax.bar_label(bars1, fmt='%.3f', padding=3); ax.bar_label(bars2, fmt='%.3f', padding=3)
plt.tight_layout()
plt.savefig('../outputs/06_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: 05_roc_curves.png, 06_model_comparison.png")

# -------------------------------------------------------
# 6. SHAP EXPLAINABILITY
# -------------------------------------------------------
print("\n[6/8] Computing SHAP values...")

best_name = results.iloc[0]['Model']
model_map = {
    'Logistic Regression': (lr_model, X_train_scaled, X_test_scaled),
    'Random Forest': (rf_model, X_train.values, X_test.values),
    'XGBoost': (xgb_model, X_train.values, X_test.values)
}
b_model, b_X_train, b_X_test = model_map[best_name]

explainer = shap.Explainer(b_model, b_X_train)
shap_values = explainer(b_X_test)
shap_test_df = pd.DataFrame(b_X_test, columns=X.columns)

plt.figure()
shap.summary_plot(shap_values, shap_test_df, show=False)
plt.title(f'SHAP Summary — {best_name}')
plt.tight_layout()
plt.savefig('../outputs/07_shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure()
shap.summary_plot(shap_values, shap_test_df, plot_type='bar', show=False)
plt.title(f'SHAP Feature Importance — {best_name}')
plt.tight_layout()
plt.savefig('../outputs/08_shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: 07_shap_summary.png, 08_shap_bar.png")

# -------------------------------------------------------
# 7. CONFUSION MATRIX
# -------------------------------------------------------
print("\n[7/8] Confusion matrix...")

best_probs_map = {'Logistic Regression': lr_probs,
                   'Random Forest': rf_probs, 'XGBoost': xgb_probs}
best_preds = (best_probs_map[best_name] >= 0.5).astype(int)
cm = confusion_matrix(y_test, best_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Non-Diabetic', 'Diabetic'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Greens', colorbar=False)
ax.set_title(f'Confusion Matrix — {best_name}')
plt.tight_layout()
plt.savefig('../outputs/09_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: 09_confusion_matrix.png")

# -------------------------------------------------------
# 8. SUMMARY
# -------------------------------------------------------
print("\n[8/8] Pipeline complete.")
print("\n" + "=" * 55)
print("  RESULTS SUMMARY")
print("=" * 55)
print(f"  Dataset:    Pima Indians Diabetes Database")
print(f"  Samples:    {df.shape[0]} | Features: {X.shape[1]}")
print(f"  Imputation: MICE (implausible zeros → NaN → imputed)")
print(f"  Split:      80/20 stratified")
print()
for _, row in results.iterrows():
    marker = " ← BEST" if row['Model'] == best_name else ""
    print(f"  {row['Model']:25s}  AUC={row['Test AUC']:.4f}{marker}")
print()
print(f"  SHAP explainability on: {best_name}")
print(f"  All outputs saved to:   ../outputs/")
print("=" * 55)
