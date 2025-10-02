#  **Customer Churn Prediction**
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/69925014-fa75-4074-9b57-9c59937cf33e" />

# TELCOVISION ANALYTICS  

## **Project Overview**  

At **TelcoVision Analytics**, our mission is to help telecom companies use data-driven strategies to keep their customers happy and reduce the chances of them leaving.  

In today's competitive telecom market, **customer churn** (when people cancel their subscriptions or switch to other providers) can seriously hurt a company's revenue and growth. Acquiring new customers can cost **5 to 7 times more** than keeping existing ones satisfied.  

Currently, many companies deal with churn **reactively**, only after customers have already left. **TelcoVision Analytics** aims to change this by using **machine learning** to predict churn before it happens, enabling businesses to step in early with **personalized offers, better support, and improved customer experiences**.  



## **Business Problem** 

Telecom providers often lack the ability to accurately identify customers at risk of leaving. This leads to:  
- Inefficient marketing spending on blanket retention campaigns  
- Missed opportunities to retain valuable customers  
- Declining revenue and reduced customer lifetime value  

**Key Challenge:**  
How can we leverage customer demographic, billing, and service usage data to accurately predict churn and proactively reduce customer attrition?  



##  **Stakeholders**  

| Stakeholder        | Role / Interest                                                             |
|--------------------|------------------------------------------------------------------------------|
| Marketing Teams    | Use churn predictions to target high-risk customers with personalized offers |
| Customer Service   | Engage with at-risk customers to resolve issues early                        |
| Senior Management  | Make strategic business decisions to reduce churn and increase revenue       |
| Data Science Team  | Build and maintain churn prediction models                                   |
| Customers          | Benefit from improved service, offers, and satisfaction                      |



## **Business Objectives**  

- Improve customer retention by identifying and intervening with high-risk customers  
- Build a predictive model to classify customers as **churn** or **non-churn**  
- Uncover the most important factors influencing churn to guide business strategy  



## **Analysis Objectives**  

- Build and evaluate a **machine learning classification model** to predict customer churn  
- Start with **Logistic Regression (baseline)**, then experiment with advanced models:  
  *Decision Trees, Random Forest, Gradient Boosting, XGBoost*  
- Identify which attributes (e.g., **tenure, monthly charges, contract type**) most strongly predict churn  
- Assess how well the models distinguish between churners and non-churners  



## **Data Understanding**  

We use the **Telco Customer Churn Dataset** from Kaggle to build our prediction model.  

### Dataset Overview  
- **Records:** 7,043 customers  
- **Features:** 21 (demographics, account info, services, churn label)  

### **Dataset Columns**  

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



##  **Data Source**  

- **Dataset Name:** Telco Customer Churn  
- **Source:** Kaggle → [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  



## **Prediction Target**  

We aim to predict the **`Churn`** variable, which indicates whether a customer has stopped using the service (`Yes`) or is still active (`No`).  

This is a **binary classification problem**. 

---

## **Exploratory Data Analysis** 

### **Univariate**  
<img width="444" height="288" alt="image" src="https://github.com/user-attachments/assets/c95f4cac-9443-419a-b748-b78098320fe1" />  
<img width="488" height="285" alt="image" src="https://github.com/user-attachments/assets/66f1e581-a666-46c5-aa48-15a6a24315e0" />  
<img width="544" height="299" alt="image" src="https://github.com/user-attachments/assets/48c33e5e-4d94-456f-9f7b-9b2684adaf95" />  

<img width="398" height="367" alt="image" src="https://github.com/user-attachments/assets/054e2ecb-5652-4c49-a204-e4cf983f2c1d" />  
<img width="395" height="343" alt="image" src="https://github.com/user-attachments/assets/fe66234a-b2d1-47d0-b031-cda7f86a731a" />  

- clean distributions, no extreme outliers  

### **Bivariate**  
<img width="395" height="333" alt="image" src="https://github.com/user-attachments/assets/eff55d36-e985-441e-a8c7-874b8c20babc" />  
<img width="382" height="278" alt="image" src="https://github.com/user-attachments/assets/ff9bc856-4f4f-4a5f-bc54-f9be84efa4bd" />  
<img width="398" height="367" alt="image" src="https://github.com/user-attachments/assets/32c59712-6074-4518-8a1f-1d473619670d" />  

- churn varies strongly by contract type, tenure and payment method  

### **Multivariate correlations**  
<img width="450" height="373" alt="image" src="https://github.com/user-attachments/assets/22dd7501-187a-4d4c-b41f-5246634ce196" />  

- Total Charges, tenure, monthly charges are highly related.  

**Key Insights:**  
- **Month-to-month contracts** and **electronic check payments** are high-risk churn factors.  
- **Long-term contracts** and **multiple bundled services** reduce churn likelihood.  



## **Feature Engineering**   
1. Removed irrelevant IDs (`customerID`).  
2. Encoded categorical variables (**OneHotEncoder**).  
3. Created new features:  
   - **AvgMonthlySpend** = TotalCharges / Tenure  
   - **NumServices** = number of subscribed services  
   - **TenureGroup** = binned tenure values  
   - **PaymentTypeSimple** = simplified payment method categories  
4. Fixed missing/invalid values.  
5. Addressed **class imbalance** (73.5% No vs. 26.5% Yes) using **SMOTE**.  



## **Modeling  Approach**

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

### Why **CatBoost?**  

<img width="380" height="288" alt="image" src="https://github.com/user-attachments/assets/cfb89965-cd89-46ba-9051-b66d3b2e5aa2" />  
<img width="395" height="288" alt="image" src="https://github.com/user-attachments/assets/4f37383b-2231-4234-9b17-aff5b740d4b9" />  

- **Highest recall (0.90)** → captures the majority of churners.  
- Lower precision is acceptable since **false negatives (missed churners)** are costlier than false positives.  
- Handles categorical variables efficiently.  


##  Key Performance Indicators (KPIs)  

- **Recall (Primary KPI)** → prioritize catching churners.  
- **Precision (Secondary KPI)** → avoid overwhelming marketing teams.  
- **ROC-AUC (Supporting KPI)** → measure overall discrimination ability.  

 **Decision** → Recall chosen as the **main KPI**, since preventing customer loss outweighs campaign inefficiency.  



## **SHAP Interpretability** 

SHAP (SHapley Additive exPlanations) was applied to the **CatBoost model** to explain predictions and highlight the most influential factors behind churn.  

### 1. SHAP Summary Plot  
<img width="554" height="667" alt="image" src="https://github.com/user-attachments/assets/982045b6-54df-4889-be5d-015fd1195ef6" />  

Top churn drivers:  
- Month-to-month contracts (strongest risk factor)  
- No online security  
- No tech support  
- Fiber optic service  

Top churn reducers:  
- Two-year contracts (most protective)  
- Longer tenure  
- Security service add-ons  

**Action Plan:**  
- Prioritize converting month-to-month customers into 1–2 year contracts.  
- Bundle online security and tech support into packages.  
- Pay special attention to fiber optic customers, who are at elevated risk.  



### 2. SHAP Feature Importance  
<img width="559" height="667" alt="image" src="https://github.com/user-attachments/assets/b514da79-fdc6-4ca3-b769-4e5e4089040b" />  

Top 6 drivers of churn (in order):  
1. Contract_Month-to-month (dominant factor)  
2. OnlineSecurity_No  
3. Contract_Two year (protective)  
4. InternetService_Fiber optic  
5. TechSupport_No  
6. Tenure  

**Key Insight:** Contract type is overwhelmingly the most impactful feature, carrying 3× more weight than any other factor.  

**Priority Actions:**  
- Target month-to-month customers with long-term contract offers.  
- Promote online security add-ons.  
- Investigate dissatisfaction among fiber optic customers.  



### 3. SHAP Force Plot (Single Prediction)  
<img width="1130" height="255" alt="image" src="https://github.com/user-attachments/assets/ff107da9-c746-42e4-9e2e-d9bf472008e6" />  

Shows individual churn predictions:  
- **High churn risk**: Fiber optic + month-to-month contracts.  
- **Low churn risk**: Two-year contracts + security and support add-ons.  

**Action:** Focus retention offers on fiber optic + month-to-month customers.  



### 4. SHAP Dependence Plot (Tenure × Monthly Charges)  
<img width="482" height="321" alt="image" src="https://github.com/user-attachments/assets/f572b25b-da91-4e46-8146-33577d83b7ad" />  

Churn risk varies by tenure:  
- **0–20 months**: Highest churn risk (danger zone).  
- **20–40 months**: Risk declines rapidly.  
- **40+ months**: Customers become stable.  

**Takeaway:** The first 20 months are critical.  
- Enhance onboarding, provide early engagement incentives, and proactively support new customers.  



### 5. SHAP Decision Plots  
<img width="667" height="586" alt="image" src="https://github.com/user-attachments/assets/1c0c6ff4-1931-43d4-bdfd-0f0e63071a03" />  
<img width="667" height="586" alt="image" src="https://github.com/user-attachments/assets/3bb90a9e-c992-456e-a025-ca6c35050d3e" />  

- **High-risk path:** Month-to-month contract → no online security → high churn probability.  
- **Low-risk path:** Two-year contract → longer tenure → bundled services → low churn.  

**Key Insight:** Contract type is the primary fork that determines churn trajectory.  



---

## **Recommendations**  

1. **Retention Strategy**  
   - Promote **long-term contracts** (offer discounts or bundles to move customers away from month-to-month).  
   - Bundle **online security and tech support** services to reduce churn risk.  
   - Provide loyalty incentives and targeted offers for **fiber optic customers**, who show higher dissatisfaction.  
   - Encourage **credit card auto-pay** or other stable payment methods instead of electronic checks.  

2. **Targeted Interventions**  
   - Deploy the CatBoost model in production to **flag high-risk customers** in real time.  
   - Launch **personalized retention campaigns** for flagged customers (e.g., discounts, contract extensions, loyalty perks).  
   - Focus retention resources on **newer customers in their first year**, as they are the most vulnerable to churn.  

3. **Business Strategy**  
   - Use churn insights to refine marketing spend: focus on the key churn drivers (contracts, services, payment methods).  
   - Provide executives with **dashboards powered by SHAP explanations** to monitor churn patterns.  
   - Continuously retrain the model with new data to adapt to evolving customer behavior.  

---

## **Final Takeaway**  

By combining a **high-recall predictive model (CatBoost)** with **SHAP interpretability**, TelcoVision Analytics can:  
- **Proactively identify customers most at risk of churn**  
- **Launch personalized retention campaigns** that directly address churn drivers  
- **Maximize ROI** by focusing resources where they will have the greatest impact  

This approach ensures that customer retention efforts are **data-driven, targeted, and effective** in reducing churn.  



## **Deployment**

The churn prediction app was deployed **locally using Streamlit** for interactive testing, visualization, and exploration.  

