# Telco Customer Churn Analysis and Prediction 📊

## 1. Project Overview 🚀

This project aims to address the critical issue of customer churn in the telecommunications industry. By analyzing historical customer data, the project identifies key factors influencing churn and develops a predictive model to forecast which customers are at risk of leaving. The insights gained are then operationalized through an interactive web application, allowing for proactive intervention and improved customer retention strategies.

## 2. Key Features ✨

* **Comprehensive Data Analysis**: In-depth Exploratory Data Analysis (EDA) and feature engineering performed on telecommunications customer data to uncover churn patterns. 🔍
* **Machine Learning Churn Prediction**: Development and training of a robust classification model to accurately predict customer churn likelihood. 🤖
* **Interactive Web Application**: A user-friendly Streamlit application for real-time churn predictions, allowing business users to input customer details and receive immediate churn forecasts. 🌐
* **Prediction Confidence Visualization**: The application provides a clear bar chart indicating the probability of a customer churning versus staying. 📊
* **Model Persistency**: The trained model is saved and loaded, enabling consistent predictions within the web application. 💾

## 3. Tech Stack Used 🛠️

This project utilizes a combination of popular data science and web development tools:

* **Programming Language**: Python 🐍
* **Data Manipulation**: `pandas`, `numpy`
* **Data Visualization**: `matplotlib`, `seaborn`, `plotly.express` 📈
* **Machine Learning**: `scikit-learn` (for model training and preprocessing)
* **Model Persistence**: `joblib`
* **Web Application Framework**: `Streamlit` 🚀
* **Development Environment**: Jupyter Notebook (`.ipynb`) 📓

## 4. How to Run the Project ▶️

Follow these steps to set up and run the project locally:

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/YourUsername/Telco-Customer-Churn-Analysis-and-Prediction.git](https://github.com/YourUsername/Telco-Customer-Churn-Analysis-and-Prediction.git)
    cd Telco-Customer-Churn-Analysis-and-Prediction
    ```
    *(Note: Replace `YourUsername` with your actual GitHub username)*

2.  **Install Dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install pandas numpy scikit-learn joblib streamlit plotly matplotlib seaborn
    ```

3.  **Model File**:
    Ensure the trained model file, `final_gb_classifier.pkl`, is present in the root directory alongside `app.py`. If you've trained your own model in the Jupyter Notebook, ensure you save it with this filename.

4.  **Run the Streamlit Application**:
    ```bash
    streamlit run app.py
    ```
    This command will open the web application in your default browser. 🌍

## 5. Model Inputs ⚙️

The Streamlit application takes the following customer features as input to make a churn prediction:

* **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents` 🧑‍🤝‍🧑
* **Account Information**: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` 💲
* **Service Information**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` 📞📺
* **Engineered Feature**: `tenure_group` (derived from `tenure`)

## 6. Insights and Outcomes 💡

* **Identified Churn Drivers**: The analysis within `Customer_Churn_Analytics.ipynb` helps pinpoint specific services, contract types, or demographic factors that strongly correlate with customer churn. 📉
* **Predictive Capability**: The trained model provides a reliable mechanism to predict customer churn, allowing businesses to anticipate and address at-risk customers. 🎯
* **Improved Retention Strategies**: By knowing which customers are likely to churn, companies can design targeted retention campaigns, personalized offers, or proactive support to minimize customer attrition. 📈
* **Enhanced Decision Making**: The interactive Streamlit app empowers non-technical stakeholders to quickly assess churn risk for individual customers or hypothetical scenarios.. ✅
