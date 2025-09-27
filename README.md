#  Customer Churn Prediction with CatBoost & Streamlit  

##  Overview  
This project focuses on predicting customer churn for a telecommunications company. Customer churn is when a customer discontinues using a company’s service. Accurately predicting churn allows businesses to implement retention strategies, reduce revenue loss and improve customer satisfaction.  

We explored data from Telco’s customer records, performed **EDA**, applied **feature engineering**, handled class imbalance, built multiple machine learning models and finally deployed the **best model (CatBoost)** using **Streamlit** for interactive predictions.  

---

##  Business and Data Understanding  

### Stakeholders  
- **Customer Retention/Marketing Teams** : need to identify at-risk customers to launch targeted retention campaigns 
- **Business Managers** : aim to reduce churn rate and improve lifetime customer value (LTV)  
- **Data Science/MLOps Teams** : responsible for model development, deployment, and monitoring  

### Dataset  
The dataset contains customer demographics, services subscribed, account information, and churn labels:  
- **Demographic features**: Gender, SeniorCitizen, Partner, Dependents  
- **Account information**: Tenure, Contract type, Payment method, Paperless billing  
- **Service features**: Internet service, Phone service, streaming services, online security  
- **Target variable**: `Churn` (Yes = customer left, No = stayed)  

Dataset size: ~7,000 records × 20+ features  

---

##  Exploratory Data Analysis (EDA)  

- **Univariate analysis** showed well-distributed numerical features with no extreme outliers
- **Bivariate analysis** revealed clear churn differences across categorical variables (contract type, tenure, payment method) 
- **Multivariate correlations** highlighted strong relationships between tenure, monthly charges, and churn 

Key insights:  
- Customers with **month-to-month contracts**, **electronic checks** and **short tenure** are more likely to churn
- Customers with **long-term contracts** and multiple services are less likely to churn

---

##  Feature Engineering  

Steps performed:  
1. Dropped irrelevant columns (`customerID`).  
2. Encoded categorical variables with **OneHotEncoder**.  
3. Engineered new features:  
   - **AvgMonthlySpend** = TotalCharges / Tenure  
   - **NumServices** = count of additional subscribed services  
   - **TenureGroup** = binned tenure values  
   - **PaymentTypeSimple** = simplified payment categories  
4. Handled missing/invalid values.  
5. Addressed **class imbalance** using **SMOTE** (73.5% No Churn vs. 26.5% Churn).  

---

##  Modeling  

We tested multiple machine learning models with pipelines including preprocessing, SMOTE and scaling (when required).  

| Model                | Tuning Method       | Notes |
|-----------------------|---------------------|-------|
| Logistic Regression   | GridSearchCV        | Baseline, interpretable |
| Decision Tree         | GridSearchCV        | Handles non-linear patterns |
| Random Forest         | RandomizedSearchCV  | Ensemble, reduces variance |
| Gradient Boosting     | GridSearchCV        | Boosting approach |
| **CatBoost** ✅        | RandomizedSearchCV  | Final best model, high recall & performance |  

### Feature Scaling  
- **Applied** for Logistic Regression (gradient-based optimization)
- **Not applied** for tree-based models (scale-invariant)  

---

## **Key Performance Indicators (KPIs)** 

The project success was measured against business-driven KPIs:  

- **Recall (Primary KPI)**  
  - Chosen because the business goal is to **catch as many churners as possible** 
  - Missing a churner (false negative) is costlier than flagging a loyal customer 
  - Ensures most at-risk customers are identified for retention

- **Precision (Secondary KPI)**  
  - Helps ensure marketing teams are not overwhelmed by false positives

- **ROC-AUC (Supporting KPI)**  
  - Evaluates overall model discrimination ability  

**Decision**: Recall was prioritized as the main KPI because in a churn context, **losing customers is more costly than extra retention efforts**

---

## Evaluation  

- Logistic Regression provided a solid **baseline**
- Tree-based models (Random Forest, Gradient Boosting) improved predictive performance  
- **CatBoost emerged as the best model**, offering the highest **recall** while maintaining good precision and ROC-AUC 
- Class imbalance was successfully mitigated with **SMOTE** improving detection of churners 

---

## SHAP Interpretability (what I ran & what it tells us)

I computed SHAP values for the final CatBoost pipeline to explain global and local predictions. The main visualizations produced:

1. **SHAP Summary Plot (dot plot)**

<p align="center">
  <img width="554" height="667" alt="SHAP summary plot" src="https://github.com/user-attachments/assets/b30b86e3-a67a-4524-b9d1-06a2305dab39" />
</p>

   - Visualizes feature impact and direction across all test samples.

2. **SHAP Feature Importance (bar plot)**

<p align="center">
  <img width="559" height="667" alt="SHAP bar plot" src="https://github.com/user-attachments/assets/42496805-e423-4138-8c15-932171cade22" />
</p>

   - Ranks features by mean absolute SHAP value.

3. **SHAP Force Plot (single sample)**

<p align="center">
  <img width="559" height="667" alt="SHAP force plot" src="https://github.com/user-attachments/assets/dd7df209-b256-4be8-9f32-c165fbbbac4d" />
</p>

   - Explains how features push a single prediction toward/away from churn.

4. **SHAP Dependence Plot** ( `tenure` vs `MonthlyCharges`)

<p align="center">
  <img width="482" height="321" alt="SHAP dependence plot" src="https://github.com/user-attachments/assets/931ce0bb-5900-4010-a1f4-efbdb387f2f8" />
</p>
  
   - Shows interaction effects and how feature values influence SHAP values.

5. **SHAP Decision Plot (global & for first 20 samples)**

<p align="center">
  <img width="667" height="586" alt="SHAP decision plot global" src="https://github.com/user-attachments/assets/2b2baf27-1d3c-4bff-91f8-e637235acfd3" />
</p>

<p align="center">
  <img width="667" height="586" alt="SHAP decision plot samples" src="https://github.com/user-attachments/assets/61c79369-b590-4f47-afdc-15f9fbdac365" />
</p>

   - Shows cumulative feature contributions per sample and groups customers into churn/retain decision paths.

**Key SHAP findings (business translation):**


- **Contract_Two year** has one of the strongest protective effects (reduces churn risk).
- **InternetService_Fiber optic** is a strong churn risk driver.
- **Contract_One year**, **tenure**, **PaymentMethod_Electronic check**, and add-on services (TechSupport, OnlineSecurity) are also influential.
- New customers (tenure < ~20 months) with high monthly charges have higher SHAP values → higher churn risk.


## Conclusion & Recommendations  

### Conclusion  
- The CatBoost model was selected as the final model and deployed via **Streamlit**  
- It delivers strong **recall**, ensuring most churners are correctly identified 
- EDA confirmed that **contract type, tenure, payment method, and additional services** are key drivers of churn 

### Recommendations  
- Focus on customers with **month-to-month contracts** : offer discounts/incentives to switch to long-term plans 
- Monitor customers paying via **electronic check** as they are more likely to churn 
- Build targeted campaigns for **new customers (<1 year tenure)** since they represent the majority of churners
- Continuously monitor model performance in production and retrain with new data to maintain accuracy 

---

##  Project Structure  


customer-churn-prediction/
├── data/                 # Raw dataset
├── notebooks/            # Jupyter notebooks with EDA, modeling, tuning
├── src/                  # Python scripts for preprocessing, modeling
├── streamlit_app.py      # Streamlit deployment file
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── models/               # Saved trained models
