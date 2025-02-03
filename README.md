# Stroke Prediction Model

## Introduction
Stroke is one of the leading causes of death worldwide, making early prediction crucial for preventive healthcare. This project leverages machine learning techniques to predict stroke risk based on demographic and medical data.

## Dataset
The model is trained using the **Stroke Prediction Dataset** from Kaggle. It includes patient attributes such as age, hypertension, heart disease, BMI, smoking status, and average glucose level.

Dataset Link: [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

### Attribute Information
- **id**: Unique identifier (removed during preprocessing)
- **gender**: Male, Female, or Other
- **age**: Patient's age
- **hypertension**: 0 (No), 1 (Yes)
- **heart_disease**: 0 (No), 1 (Yes)
- **ever_married**: Yes/No
- **work_type**: Type of employment
- **Residence_type**: Rural/Urban
- **avg_glucose_level**: Average blood glucose level
- **bmi**: Body Mass Index (BMI)
- **smoking_status**: Formerly smoked, never smoked, smokes, or unknown
- **stroke**: 1 (Had a stroke), 0 (No stroke)

## Data Preprocessing
- Missing BMI values were imputed using a **Decision Tree Regressor**.
- Class imbalance was addressed by **equalizing stroke and non-stroke cases**.
- Categorical variables were **encoded into numerical values**.

## Machine Learning Models
Two models were implemented and evaluated:
- **Logistic Regression**
- **Decision Tree Classifier**

### Model Performance
| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 70%      |
| Decision Tree      | 60%      |

Feature importance analysis revealed that **age, BMI, and glucose level** were the most critical factors affecting stroke risk.

## Feature Importance
| Feature           | Importance |
|------------------|------------|
| Age             | 0.43       |
| BMI             | 0.23       |
| Glucose Level   | 0.21       |
| Heart Disease   | 0.02       |

## Usage
### Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Running the Model
```python
import joblib
import pandas as pd
import numpy as np

# Load the model
model = joblib.load("log_reg_model.pkl")

# Example patient data
new_patient = pd.DataFrame({
    "gender": [0],
    "age": [70],
    "hypertension": [0],
    "heart_disease": [0],
    "ever_married": [0],
    "work_type": [1],
    "Residence_type": [1],
    "avg_glucose_level": [80],
    "bmi": [15],
    "smoking_status": [0],
})

# Predict stroke risk
prediction = model.predict(new_patient)
probability = model.predict_proba(new_patient)
print(f"Predicted Stroke Risk: {prediction}")
print(f"Probability Distribution: {probability}")
```

## Future Improvements
- Train the model with a **larger dataset** for better generalization.
- Implement **advanced algorithms** (e.g., Random Forest, XGBoost).
- Include **additional medical history** for more accurate predictions.

## Author
Developed by **Çağatay Elik**

---
This project serves as a valuable tool for early stroke risk assessment, contributing to **preventive healthcare and clinical decision-making**.

