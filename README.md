# Diabetes Risk Prediction Using Machine Learning

An end-to-end machine learning pipeline for predicting diabetes risk using clinical features, with MICE imputation and SHAP explainability.

**Author:** Chantale Jepleting Ruto  
**MSc Health Informatics (Data Science Option) — Moi University, Eldoret, Kenya**

---

## Project Overview

Diabetes mellitus is a growing public health crisis globally, with Sub-Saharan Africa carrying a rapidly increasing burden. According to the IDF Diabetes Atlas, an estimated 24 million adults in Africa were living with diabetes in 2021, projected to reach 55 million by 2045 — yet most remain undiagnosed due to limited screening infrastructure.

This project builds and compares three supervised ML models to predict diabetes risk from routine clinical measurements, with a focus on features that are realistically available in low-resource settings such as Kenya. SHAP values are used to make model predictions interpretable for potential clinical use.

> **Note on dataset:** This project uses the Pima Indians Diabetes Database — one of the few publicly available, well-validated diabetes datasets with appropriate clinical features. A significant limitation in African diabetes ML research is the near-complete absence of open-access clinical datasets from the continent. This gap is itself a research and advocacy priority. Future work will seek to replicate this pipeline on African cohort data (e.g. AWI-Gen, AMPATH) once accessible.

---

## Dataset

**Pima Indians Diabetes Database**  
- 768 patients | 8 clinical features | Binary outcome (1 = Diabetic, 0 = Non-Diabetic)  
- Class distribution: ~65% non-diabetic, ~35% diabetic  
- Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/diabetes) / [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

**Features:**

| Feature | Description | LMIC relevance |
|---|---|---|
| Pregnancies | Number of pregnancies | ✅ Routinely recorded |
| Glucose | Plasma glucose (mg/dL) | ✅ Basic lab test |
| BloodPressure | Diastolic BP (mmHg) | ✅ Routinely measured |
| SkinThickness | Triceps skinfold (mm) | ⚠️ Less common in LMICs |
| Insulin | 2-hour serum insulin | ⚠️ Often unavailable |
| BMI | Body mass index | ✅ Routinely measured |
| DiabetesPedigreeFunction | Family history score | ✅ Collectable via history |
| Age | Patient age (years) | ✅ Always available |

---

## Methods

| Step | Approach |
|---|---|
| Zero-value handling | Implausible zeros (Glucose, BP, BMI, etc.) replaced with NaN |
| Missing data | MICE (Multivariate Imputation by Chained Equations) |
| Class imbalance | `class_weight='balanced'` / `scale_pos_weight` |
| Train/Test split | 80/20, stratified |
| Feature scaling | StandardScaler (Logistic Regression only) |
| Cross-validation | 5-fold, AUC metric |
| Explainability | SHAP values on best model |

**Models compared:**
- Logistic Regression
- Random Forest (200 trees)
- XGBoost (200 estimators)

---

## Results

| Model | Test AUC | CV AUC (5-fold) |
|---|---|---|
| Logistic Regression | 0.8120 ← Best | 0.8417 |
| Random Forest | 0.8092 | 0.8312 |
| XGBoost | 0.7944 | 0.7784 |

All models met or approached the AUC ≥ 0.80 clinical prediction threshold. Logistic Regression achieved the best generalisation, consistent with literature on this dataset.

**SHAP finding:** Glucose was the strongest predictor of diabetes risk, followed by BMI and Age — consistent with clinical knowledge and published literature.

---

## Repository Structure

```
diabetes-risk-ml/
├── data/
│   └── diabetes.csv                      # Pima Indians Diabetes Dataset
├── notebooks/
│   └── diabetes_risk_prediction.py       # Full ML pipeline
├── outputs/
│   ├── 01_class_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_feature_distributions.png
│   ├── 04_boxplots.png
│   ├── 05_roc_curves.png
│   ├── 06_model_comparison.png
│   ├── 07_shap_summary.png
│   ├── 08_shap_bar.png
│   └── 09_confusion_matrix.png
└── README.md
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/ChantaleRuto/diabetes-risk-ml.git
cd diabetes-risk-ml

# Install dependencies
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn

# Run the pipeline
cd notebooks
python diabetes_risk_prediction.py
```

---

## Key References

- IDF Diabetes Atlas, 10th Edition (2021). International Diabetes Federation.
- Smith, J.W. et al. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Symposium on Computer Applications and Medical Care.*
- Adda, R.B. (2025). Improving Type 2 Diabetes Prediction using ML from the AWI-Gen Cohort, Ghana. AUC 0.845 with XGBoost.

---

## Related Work

This portfolio project complements my MSc thesis research:  
**"Predicting Liver Disease Progression Using Machine Learning: A Case Study of the MTRH Gastrointestinal Clinic"**  
*DS-I Africa Early Career Researcher Fellow (NIH Fogarty International Center)*  
*Moi University / AMPATH — Eldoret, Kenya*
