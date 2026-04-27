# Bank Customer Churn Prediction

## Project Overview
This project predicts whether a banking customer is likely to leave the bank (`Exited = 1`) using historical customer profile and account data. The objective is to help a bank identify high-risk customer segments, understand churn drivers, and support data-driven retention strategies.

## Business Problem
Customer churn reduces revenue and increases customer acquisition costs. By predicting churn risk early, banks can target vulnerable customers with retention actions such as personalized offers, relationship manager follow-ups, product education, or improved service support.

## Dataset
File used: `data/raw/Churn_Modelling.csv`

Typical columns include:
- Customer profile: `Geography`, `Gender`, `Age`
- Banking relationship: `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`
- Financial indicator: `CreditScore`, `EstimatedSalary`
- Target variable: `Exited`

## Project Pipeline
1. **Data Understanding**
   - Load dataset
   - Review shape, missing values, data types, and target distribution

2. **Data Cleaning**
   - Remove non-predictive identifiers: `RowNumber`, `CustomerId`, `Surname`
   - Check missing values and duplicates
   - Separate features and target variable

3. **Exploratory Data Analysis**
   - Churn distribution
   - Churn by geography and gender
   - Age, balance, activity, and product-based churn patterns
   - Correlation analysis for numeric variables

4. **Feature Engineering and Preprocessing**
   - One-hot encode categorical features
   - Scale numerical features
   - Split data into train/test sets
   - Use a preprocessing pipeline to prevent data leakage

5. **Model Development**
   - Train Logistic Regression, Random Forest, and Gradient Boosting models
   - Compare performance using Accuracy, Precision, Recall, F1-score, and ROC-AUC
   - Select the best model based on ROC-AUC and recall balance

6. **Model Evaluation**
   - Classification report
   - Confusion matrix
   - ROC-AUC score
   - Feature importance chart where supported

7. **Deployment-Ready Prediction**
   - Save trained model as `models/best_model.pkl`
   - Use `src/predict.py` for customer-level prediction
   - Use `app/streamlit_app.py` for an interactive dashboard

## Folder Structure
```text
bank_customer_churn_prediction/
│
├── data/
│   ├── raw/
│   │   └── Churn_Modelling.csv
│   └── processed/
│
├── notebooks/
│   └── 01_bank_churn_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── app/
│   └── streamlit_app.py
│
├── models/
├── reports/
│   └── figures/
├── outputs/
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run the Project

### 1. Create and activate a virtual environment
```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

### 2. Install required packages
```bash
pip install -r requirements.txt
```

### 3. Train the models
```bash
python src/train_model.py
```

### 4. Evaluate the saved model
```bash
python src/evaluate_model.py
```

### 5. Predict churn for a sample customer
```bash
python src/predict.py
```

### 6. Run the Streamlit app
```bash
streamlit run app/streamlit_app.py
```

## Resume-Friendly Project Description
Built an end-to-end banking customer churn prediction project using Python, pandas, scikit-learn, and Streamlit. Developed a complete machine learning pipeline including data cleaning, exploratory analysis, preprocessing, model comparison, performance evaluation, churn risk prediction, and dashboard deployment. The project identifies customers vulnerable to attrition and supports targeted retention strategies using interpretable model outputs.

## Tools Used
- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- joblib
- Streamlit

## Key Business Value
The model helps banking teams prioritize retention campaigns by identifying customers with high churn probability, reducing avoidable attrition and improving customer lifetime value.
