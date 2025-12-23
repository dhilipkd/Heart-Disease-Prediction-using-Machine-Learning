# Heart-Disease-Prediction-using-Machine-Learning
# Heart Disease Risk Prediction using Machine Learning

This project builds an **end-to-end Heart Disease Risk Prediction system** using Machine Learning classification techniques.  
Multiple models are trained, tuned, evaluated, and the best-performing model is deployed using **Streamlit** for real-time risk prediction.

## Project Overview

Heart disease is one of the leading causes of death worldwide.  
This project predicts whether a patient is at **high risk or low risk of heart disease** based on demographic, lifestyle, and medical features using supervised machine learning models.

## Key Features

- End-to-end ML pipeline (data cleaning → deployment)
- Removal of low-impact and redundant medical features
- Handling missing values using **SimpleImputer**
- Scaling numerical features using **StandardScaler**
- Encoding categorical features using **OneHotEncoder**
- Model training using:
  - Logistic Regression  
  - Random Forest Classifier  
  - XGBoost Classifier
- Hyperparameter tuning using **GridSearchCV**
- Model evaluation using:
  - Accuracy  
  - ROC AUC Score  
  - Confusion Matrix  
  - Classification Report
- Best model selection based on ROC AUC
- Saving trained model as `.pkl`
- Interactive **Streamlit Web App** for prediction

## Machine Learning Workflow

1. Load and inspect heart disease dataset  
2. Drop redundant and low-impact features  
3. Handle missing values  
4. Encode target variable  
5. Split data into features and target  
6. Build preprocessing pipeline using `ColumnTransformer`  
7. Train multiple classification models  
8. Perform hyperparameter tuning using GridSearchCV  
9. Evaluate models using ROC AUC and accuracy  
10. Save the best performing model  
11. Deploy the model using Streamlit  

## Tech Stack

- **Language:** Python  
- **Libraries:**  
  - Pandas, NumPy  
  - Scikit-learn  
  - XGBoost  
  - Matplotlib, Seaborn  
  - Joblib  
- **Deployment:** Streamlit  

## Model Evaluation Metrics

Models are evaluated using healthcare-relevant metrics:

- **Accuracy** – Overall correctness of predictions  
- **ROC AUC Score** – Ability to distinguish high-risk vs low-risk patients  
- **Confusion Matrix** – True vs False predictions  
- **Classification Report** – Precision, Recall, F1-score  

The **best model is selected based on ROC AUC score**.

## Model Saving
```python
import joblib
joblib.dump(best_model, "best_model.pkl")
```

# Streamlit Web Application
The Streamlit app allows users to:
- Enter patient demographic details
- Add lifestyle and medical history information
- Predict heart disease risk instantly
- View probability score for better decision support

Input Categories:
Patient Information
- Age
- Gender
- Blood Pressure
- Cholesterol Level
- BMI
- Sleep Hours

Lifestyle Factors
- Exercise Habits
- Smoking
- Alcohol Consumption
- Stress Level

Medical History
- Family Heart Disease
- Diabetes
- Triglyceride Level
- Fasting Blood Sugar

## Results
- Displays:
    - Best Model Name
    - Best ROC AUC Score
    - Model comparison summary
- Prediction output:
    - High Risk or Low Risk
    - Risk probability score

## Author
Dhilip K
