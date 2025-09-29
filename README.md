#  Customer Churn Prediction
# TELCOVISION ANALYTICS  

## Project Overview  

At **TelcoVision Analytics**, our mission is to help telecom companies use data-driven strategies to keep their customers happy and reduce the chances of them leaving.  

In today's competitive telecom market, **customer churn** (when people cancel their subscriptions or switch to other providers) can seriously hurt a company's revenue and growth. Acquiring new customers can cost **5 to 7 times more** than keeping existing ones satisfied.  

Currently, many companies deal with churn **reactively**, only after customers have already left. **TelcoVision Analytics** aims to change this by using **machine learning** to predict churn before it happens, enabling businesses to step in early with **personalized offers, better support, and improved customer experiences**.  

---

## Business Problem  

Telecom providers often lack the ability to accurately identify customers at risk of leaving. This leads to:  
- Inefficient marketing spending on blanket retention campaigns  
- Missed opportunities to retain valuable customers  
- Declining revenue and reduced customer lifetime value  

**Key Challenge:**  
How can we leverage customer demographic, billing, and service usage data to accurately predict churn and proactively reduce customer attrition?  

---

##  Stakeholders  

| Stakeholder        | Role / Interest                                                             |
|--------------------|------------------------------------------------------------------------------|
| Marketing Teams    | Use churn predictions to target high-risk customers with personalized offers |
| Customer Service   | Engage with at-risk customers to resolve issues early                        |
| Senior Management  | Make strategic business decisions to reduce churn and increase revenue       |
| Data Science Team  | Build and maintain churn prediction models                                   |
| Customers          | Benefit from improved service, offers, and satisfaction                      |

---

## Business Objectives  

- Improve customer retention by identifying and intervening with high-risk customers  
- Build a predictive model to classify customers as **churn** or **non-churn**  
- Uncover the most important factors influencing churn to guide business strategy  

---

## Analysis Objectives  

- Build and evaluate a **machine learning classification model** to predict customer churn  
- Start with **Logistic Regression (baseline)**, then experiment with advanced models:  
  *Decision Trees, Random Forest, Gradient Boosting, XGBoost*  
- Identify which attributes (e.g., **tenure, monthly charges, contract type**) most strongly predict churn  
- Assess how well the models distinguish between churners and non-churners  

---

## Data Understanding  

We use the **Telco Customer Churn Dataset** from Kaggle to build our prediction model.  

### Dataset Overview  
- **Records:** 7,043 customers  
- **Features:** 21 (demographics, account info, services, churn label)  

### Dataset Columns  

| Column Name        | Description                                                    | Data Type                |
|--------------------|----------------------------------------------------------------|--------------------------|
| customerID         | Unique identifier for each customer                            | Categorical (ID)         |
| gender             | Gender of the customer                                         | Categorical              |
| SeniorCitizen      | Indicates if the customer is a senior citizen (1 = Yes, 0 = No)| Numeric (Binary)         |
| Partner            | Whether the customer has a partner                             | Categorical              |
| Dependents         | Whether the customer has dependents                            | Categorical              |
| tenure             | Number of months the customer has stayed                       | Numeric (Discrete)       |
| PhoneService       | Whether the customer has phone service                         | Categorical              |
| MultipleLines      | Whether the customer has multiple phone lines                  | Categorical              |
| InternetService    | Internet type (DSL, Fiber optic, None)                         | Categorical              |
| OnlineSecurity     | Whether the customer has online security                       | Categorical              |
| OnlineBackup       | Whether the customer has online backup                         | Categorical              |
| DeviceProtection   | Whether the customer has device protection                     | Categorical              |
| TechSupport        | Whether the customer has tech support                          | Categorical              |
| StreamingTV        | Whether the customer has streaming TV                          | Categorical              |
| StreamingMovies    | Whether the customer has streaming movies                      | Categorical              |
| Contract           | Contract type (Month-to-month, One year, Two year)             | Categorical              |
| PaperlessBilling   | Whether the customer uses paperless billing                    | Categorical              |
| PaymentMethod      | Customer’s payment method                                      | Categorical              |
| MonthlyCharges     | Amount charged monthly                                         | Numeric (Continuous)     |
| TotalCharges       | Total amount billed                                            | Numeric (Continuous)     |
| Churn              | Target: Whether the customer left (Yes = churn, No = stayed)   | Categorical (Target)     |

---

##  Data Source  

- **Dataset Name:** Telco Customer Churn  
- **Source:** Kaggle → [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  

---

## Prediction Target  

We aim to predict the **`Churn`** variable, which indicates whether a customer has stopped using the service (`Yes`) or is still active (`No`).  

This is a **binary classification problem**. 


## Exploratory Data Analysis (EDA)  

- **Univariate**  

<p align="center">
  <img width="398" height="367" alt="image" src="https://github.com/user-attachments/assets/054e2ecb-5652-4c49-a204-e4cf983f2c1d" />  
  <img width="395" height="343" alt="image" src="https://github.com/user-attachments/assets/fe66234a-b2d1-47d0-b031-cda7f86a731a" />  
</p>  

- clean distributions, no extreme outliers  

- **Bivariate**  

<p align="center">
  <img width="395" height="333" alt="image" src="https://github.com/user-attachments/assets/eff55d36-e985-441e-a8c7-874b8c20babc" />  
  <img width="382" height="278" alt="image" src="https://github.com/user-attachments/assets/ff9bc856-4f4f-4a5f-bc54-f9be84efa4bd" />  
  <img width="398" height="367" alt="image" src="https://github.com/user-attachments/assets/32c59712-6074-4518-8a1f-1d473619670d" />  
</p>  

- churn varies strongly by contract type, tenure and payment method  

- **Multivariate correlations**  

<p align="center">
  <img width="450" height="373" alt="image" src="https://github.com/user-attachments/assets/22dd7501-187a-4d4c-b41f-5246634ce196" />  
</p>  

- Total Charges, tenure, monthly charges are highly related.  

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

<p align="center">
  <img width="380" height="288" alt="image" src="https://github.com/user-attachments/assets/cfb89965-cd89-46ba-9051-b66d3b2e5aa2" />  
  <img width="395" height="288" alt="image" src="https://github.com/user-attachments/assets/4f37383b-2231-4234-9b17-aff5b740d4b9" />  
</p>  

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

<p align="center">
  <img width="1130" height="255" alt="image" src="https://github.com/user-attachments/assets/44d3c7bf-2cca-494b-b54a-cc3ef80d7a83" />  
</p>  

SHAP (SHapley Additive exPlanations) was applied to the **CatBoost model** for interpretability.  

### 1. SHAP Summary Plot  
<p align="center">  
  <img width="554" src="https://github.com/user-attachments/assets/b30b86e3-a67a-4524-b9d1-06a2305dab39" />  
</p>  

### 2. SHAP Feature Importance  
<p align="center">  
  <img width="559" src="https://github.com/user-attachments/assets/42496805-e423-4138-8c15-932171cade22" />  
</p>  

### 3. SHAP Force Plot (Single Sample)  
<p align="center">  
  <img width="559" src="https://github.com/user-attachments/assets/dd7df209-b256-4be8-9f32-c165fbbbac4d" />  
</p>  

### 4. SHAP Dependence Plot (Tenure × MonthlyCharges)  
<p align="center">  
  <img width="482" src="https://github.com/user-attachments/assets/931ce0bb-5900-4010-a1f4-efbdb387f2f8" />  
</p>  

### 5. SHAP Decision Plots  
<p align="center">  
  <img width="667" src="https://github.com/user-attachments/assets/2b2baf27-1d3c-4bff-91f8-e637235acfd3" />  
</p>  
<p align="center">  
  <img width="667" src="https://github.com/user-attachments/assets/61c79369-b590-4f47-afdc-15f9fbdac365" />  
</p>  

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
