#  Customer Churn Prediction with CatBoost & Streamlit  

##  Overview  
This project predicts customer churn for a telecommunications company.  
Churn occurs when customers discontinue services, leading to significant revenue loss.  

By predicting churn, businesses can:  
- implement targeted retention strategies,  
- reduce customer acquisition costs, and  
- improve lifetime value (LTV).  

We performed **EDA**, **feature engineering**, **class imbalance handling**, tested **7 machine learning models**, and finally deployed the **best model (CatBoost)** with **Streamlit** for interactive predictions and interpretability.  

---

##  Business & Data Understanding  

### Stakeholders  
- **Customer Retention/Marketing Teams** → need to identify at-risk customers for campaigns.  
- **Business Managers** → aim to reduce churn rate and improve profitability.  
- **Data Science/MLOps Teams** → responsible for model building, deployment, and monitoring.  

### Dataset  
Contains customer demographics, services, account details, and churn labels.  

- **Demographics** → Gender, SeniorCitizen, Partner, Dependents  
- **Account Info** → Tenure, Contract type, Payment method, Paperless billing  
- **Services** → Internet, Phone, Streaming, Security  
- **Target** → `Churn` (Yes = left, No = stayed)  

**Size**: ~7,000 customers × 20+ features  

---

## Exploratory Data Analysis (EDA)  

- **Univariate**
 <img width="398" height="367" alt="image" src="https://github.com/user-attachments/assets/054e2ecb-5652-4c49-a204-e4cf983f2c1d" />
 <img width="395" height="343" alt="image" src="https://github.com/user-attachments/assets/fe66234a-b2d1-47d0-b031-cda7f86a731a" />

 - clean distributions, no extreme outliers


- **Bivariate**
  <img width="395" height="333" alt="image" src="https://github.com/user-attachments/assets/eff55d36-e985-441e-a8c7-874b8c20babc" />

<img width="382" height="278" alt="image" src="https://github.com/user-attachments/assets/ff9bc856-4f4f-4a5f-bc54-f9be84efa4bd" />

<img width="398" height="367" alt="image" src="https://github.com/user-attachments/assets/32c59712-6074-4518-8a1f-1d473619670d" />

- churn varies strongly by contract type, tenure and payment method
- **Multivariate correlations**
 <img width="450" height="373" alt="image" src="https://github.com/user-attachments/assets/22dd7501-187a-4d4c-b41f-5246634ce196" />


- tenure, monthly charges, and churn are highly related.  

**Key Insights:**  
- **Month-to-month contracts** and **electronic check payments** are high-risk churn factors.  
- **Long-term contracts** and **multiple bundled services** reduce churn likelihood.  

---

## Feature Engineering  

Steps applied:  
1. Removed irrelevant IDs (`customerID`).  
2. Encoded categorical variables (**OneHotEncoder**).  
3. Created new features:  
   - **AvgMonthlySpend** = TotalCharges / Tenure  
   - **NumServices** = number of subscribed services  
   - **TenureGroup** = binned tenure values  
   - **PaymentTypeSimple** = simplified payment method categories  
4. Fixed missing/invalid values.  
5. Addressed **class imbalance** (73.5% No vs. 26.5% Yes) using **SMOTE**.  

---

## Modeling  

We evaluated **7 models** using preprocessing pipelines + SMOTE + hyperparameter tuning.  

| Model                | Method              | Recall | Precision | Notes |
|-----------------------|---------------------|--------|-----------|-------|
| Logistic Regression   | GridSearchCV        | 0.78   | 0.50      | Baseline, interpretable |
| Decision Tree         | GridSearchCV        | 0.71   | 0.48      | Non-linear splits |
| Random Forest         | RandomizedSearchCV  | 0.67   | 0.56      | Strong ensemble |
| Gradient Boosting     | GridSearchCV        | 0.64   | 0.60      | Boosting, moderate performance |
| XGBoost               | RandomizedSearchCV  | 0.65   | 0.59      | Advanced boosting |
| LightGBM              | GridSearchCV        | 0.69   | 0.55      | Fast boosting, balanced |
| **CatBoost ✅**        | RandomizedSearchCV  | **0.90** | 0.44 | Best recall, selected final model |

### Why CatBoost?  
- **Highest recall (0.90)** → captures the majority of churners.  
- Lower precision is acceptable since **false negatives (missed churners)** are costlier than false positives.  
- Handles categorical variables efficiently.  

---

##  Key Performance Indicators (KPIs)  

- **Recall (Primary KPI)** → prioritize catching churners.  
- **Precision (Secondary KPI)** → avoid overwhelming marketing teams.  
- **ROC-AUC (Supporting KPI)** → measure overall discrimination ability.  

 **Decision** → Recall chosen as the **main KPI**, since preventing customer loss outweighs campaign inefficiency.  

---

##  SHAP Interpretability  

SHAP (SHapley Additive exPlanations) was applied to the **CatBoost model** for interpretability.  

### 1. SHAP Summary Plot  
<p align="center">  
  <img width="554" src="https://github.com/user-attachments/assets/b30b86e3-a67a-4524-b9d1-06a2305dab39" />  
</p>  
Shows the overall impact and direction of each feature.  

### 2. SHAP Feature Importance  
<p align="center">  
  <img width="559" src="https://github.com/user-attachments/assets/42496805-e423-4138-8c15-932171cade22" />  
</p>  
Ranks features by their average contribution to churn predictions.  

### 3. SHAP Force Plot (Single Sample)  
<p align="center">  
  <img width="559" src="https://github.com/user-attachments/assets/dd7df209-b256-4be8-9f32-c165fbbbac4d" />  
</p>  
Explains how individual features push a single prediction toward churn or retention.  

### 4. SHAP Dependence Plot (Tenure × MonthlyCharges)  
<p align="center">  
  <img width="482" src="https://github.com/user-attachments/assets/931ce0bb-5900-4010-a1f4-efbdb387f2f8" />  
</p>  
Shows new customers (<20 months) with high charges are at **highest churn risk**.  

### 5. SHAP Decision Plots  
<p align="center">  
  <img width="667" src="https://github.com/user-attachments/assets/2b2baf27-1d3c-4bff-91f8-e637235acfd3" />  
</p>  
<p align="center">  
  <img width="667" src="https://github.com/user-attachments/assets/61c79369-b590-4f47-afdc-15f9fbdac365" />  
</p>  
Visualizes decision paths: churners (red, right) vs retainers (blue, left).  

---

###  Key SHAP Insights  
- **Contract_Two year** strongly reduces churn risk.  
- **InternetService_Fiber optic** increases churn risk.  
- **Electronic check payments** are associated with churn.  
- **Short-tenure, high-paying customers** are at highest risk.  

---

## Conclusion & Recommendations  

### Conclusion  
- **CatBoost** is the final deployed model with highest recall.  
- Key churn drivers: **contract type, tenure, payment method, add-on services**.  
- SHAP confirms business logic with transparent feature influence.  

### Recommendations  
- Incentivize **month-to-month customers** to switch to long-term contracts
