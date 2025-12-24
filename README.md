# 🩺 Diabetes Risk Prediction — ML + Web Integration (Student Project)

This is a **BTech CSE student project (3rd semester completed)** that explores diabetes risk prediction using structured ML workflow and a simple Flask web interface.

> ⚠️ **Educational purpose only. Not medical advice.**  
> Built for learning ML fundamentals, feature engineering, model evaluation, and deployment ethics.

---

## 🧠 Feature Engineering
The following engineered features were created to improve predictive signals:

| Feature | Description |
|--------|-------------|
| `BMI_scaled`, `Glucose_scaled` | Normalized using **MinMaxScaler** for better model stability |
| `Metabolic_risk` | Custom risk index: `0.6 × Glucose + 0.4 × BMI` |
| Zero value imputation | Medical nulls disguised as 0 replaced with **median imputation** |

Feature engineering experiments showed that **including `Metabolic_risk` improved recall and ROC-AUC** compared to training without it.

---

## 🤖 All Models Trained for Comparison
Multiple models were trained to compare recall performance and ROC-AUC behavior:

| Model | Type | Why it was used |
|------|------|----------------|
| Logistic Regression | Linear classifier | Baseline medical risk benchmark |
| Linear SVC | Margin-based | To compare hinge loss vs probabilistic methods |
| Random Forest | Tree ensemble | Final selected model due to best recall & ROC-AUC |
| Voting Classifier | (Logistic Regression + Linear SVC + Random Forest) | To compare overfitting vs ensemble stability |





---

## 📊 Model Evaluation Focus
Since this is a medical-domain dataset, evaluation prioritized:

- **High Recall** → reduce **False Negatives**
- **ROC-AUC** → ensure strong class separation
- **Class imbalance handling** using **Stratified split**

### Observed performance:
- Random Forest **with Metabolic Risk** → **Recall 0.77**
- Random Forest **without Metabolic Risk** → **Recall 0.74**

> 📌 Including engineered feature helped, and **no snooping bias was introduced**, as no test-set tuning was done.

---

## 📓 Jupyter Notebook Experiments
Different notebooks created during project workflow:

