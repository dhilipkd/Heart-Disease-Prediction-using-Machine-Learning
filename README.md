# Heart-Disease-Prediction-using-Machine-Learning
This project builds a complete Machine Learning system to predict heart disease using Logistic Regression, Random Forest, and XGBoost. The workflow includes data preprocessing, feature engineering, model training, hyperparameter tuning, performance comparison, and saving the best model.

## Features
- Handles missing values, encoding, and scaling using Scikit-Learn Pipelines  
- Trains three classification models  
- Hyperparameter tuning with GridSearchCV  
- Evaluates models using Accuracy, ROC-AUC, Confusion Matrix, and Classification Report  
- Saves the best-performing model as a .pkl file  
- Visualizes confusion matrices for all models

## Tech Stack
Python, Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn, Joblib

## Project Workflow
1. Load and explore dataset  
2. Preprocess numerical and categorical features  
3. Build ML pipelines for multiple classifiers  
4. Perform hyperparameter tuning  
5. Compare performance using ROC-AUC and Accuracy  
6. Save the best model using Joblib  

<img width="695" height="816" alt="image" src="https://github.com/user-attachments/assets/8bd1db27-b419-4bb0-847a-20cba4581950" />

<img width="730" height="817" alt="image" src="https://github.com/user-attachments/assets/072e9afe-aa9b-43b0-a005-2a03419dd550" />

<img width="977" height="835" alt="image" src="https://github.com/user-attachments/assets/a898310f-bb59-4bed-83f5-cfabad7813d1" />


## Model Saving
```python
import joblib
joblib.dump(best_model, "best_model.pkl")
```


## Results
The script prints:
- Best model based on ROC-AUC  
- Model comparison summary (Accuracy + ROC-AUC)  
- Confusion Matrix visualizations  

