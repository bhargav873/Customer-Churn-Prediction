# Customer Churn Prediction

## Project Overview

Customer Churn Prediction is a machine learning project that identifies customers who are likely to discontinue a company's services.

The project uses customer demographic information, account details, subscription plans, and service usage data to analyze churn patterns and predict the likelihood of customer churn.

The goal is to help businesses understand customer behavior, identify customers who may leave, and develop data-driven customer retention strategies.

## Objectives

* Analyze customer data to understand churn patterns.
* Clean and prepare data for machine learning.
* Perform Exploratory Data Analysis (EDA).
* Train and compare multiple machine learning models.
* Evaluate model performance using classification metrics.
* Compare prediction thresholds to understand the precision-recall trade-off.
* Predict churn probability for an individual customer.
* Generate business insights and customer retention recommendations.
* Save the trained machine learning pipeline for future use.

## Dataset

**Dataset:** Telco Customer Churn

The dataset contains customer information from a telecommunications company.

It includes details such as:

* Customer demographics
* Tenure and account information
* Internet and phone services
* Contract types
* Payment methods
* Monthly charges
* Total charges
* Customer churn status

**Target variable:** `Churn`

* `Yes` — The customer has churned.
* `No` — The customer has not churned.

The dataset contains approximately 7,043 customer records and 21 original columns.

Place the dataset file in the project root directory using this filename:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Jupyter Notebook / PyCharm

## Project Workflow

### 1. Data Loading and Exploration

* Loaded the dataset using Pandas.
* Examined the dataset structure, column names, and data types.
* Checked for missing values and duplicate records.

### 2. Data Cleaning and Preprocessing

* Removed duplicate records.
* Converted `TotalCharges` into a numeric data type.
* Handled missing numeric values using median imputation in the machine learning pipeline.
* Converted categorical variables into numerical features using one-hot encoding.
* Scaled numerical features where appropriate.

### 3. Exploratory Data Analysis

Analyzed customer churn patterns using visualizations and grouped summaries.

The analysis examined:

* Overall churn distribution.
* Churn across different contract types.
* Churn across customer tenure groups.
* Churn across internet service categories.

### 4. Machine Learning Models

Trained and compared three classification algorithms:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

### 5. Model Evaluation

Evaluated model performance using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

Compared the models to understand their performance in identifying customers who churn.

### 6. Prediction Threshold Analysis

Compared classification thresholds of 0.30, 0.40, and 0.50 using predicted churn probabilities.

This analysis demonstrates how changing the threshold affects precision and recall, helping illustrate the trade-off between identifying more potential churners and reducing false alarms.

### 7. Business Insights

The analysis identified several patterns:

* Customers on month-to-month contracts had a higher churn rate than customers on longer-term contracts.
* Customers with shorter tenure showed a higher churn rate.
* Customers using fiber-optic internet had a higher churn rate than customers using DSL or no internet service.

These findings indicate areas businesses could investigate when developing customer retention strategies. They describe associations in the dataset and do not establish the causes of churn.

### 8. Individual Customer Prediction

The project accepts an example customer's details and estimates the probability that the customer will churn.

The prediction is based on the trained Random Forest model and a selected classification threshold.

The probability is a model estimate, not a guarantee of future customer behavior.

### 9. Model Saving

Saved the trained Random Forest pipeline using Joblib for potential reuse.

Saved model location:

`saved_model/customer_churn_model.pkl`

The saved pipeline includes the preprocessing steps and trained model.

## Model Comparison

The following results were obtained during the initial model comparison:

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |    80.7% |     65.8% |  56.7% |    60.9% |
| Decision Tree       |    73.0% |     49.1% |  49.2% |    49.1% |
| Random Forest       |    77.6% |     56.8% |  65.5% |    60.9% |

The Random Forest model achieved higher recall than Logistic Regression in this comparison, meaning it identified a larger proportion of actual churners.

Logistic Regression achieved higher accuracy and precision in the same comparison.

The most suitable model depends on the business objective and the relative costs of false positives and false negatives.

*Note: These are preliminary results from the current workflow. The final model and threshold should be evaluated using a properly separated validation/test process before making deployment or business-performance claims.*

## Business Recommendations

Based on the observed churn patterns, a business could consider:

1. Investigating retention strategies for month-to-month contract customers.
2. Improving onboarding and support for newly acquired customers.
3. Examining customer satisfaction, pricing, and service quality among fiber-optic internet customers.
4. Monitoring churn-related metrics regularly.
5. Using model predictions as one input to customer retention decisions rather than treating them as definitive outcomes.

## Project Structure

```text
Customer Churn Prediction/
│
├── main.py
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── README.md
├── requirements.txt
├── .gitignore
│
└── saved_model/
    └── customer_churn_model.pkl
```

The `.venv/` directory is excluded from version control.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the Project Folder

```bash
cd "Customer Churn Prediction"
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Project

```bash
python main.py
```

Ensure the dataset CSV is present in the project root before running the script.

## Future Improvements

* Use cross-validation and a dedicated validation set for more reliable model selection and threshold tuning.
* Improve model performance through hyperparameter tuning.
* Add a Streamlit interface for interactive predictions.
* Track model performance on new customer data.
* Explore additional features and customer retention strategies.

## Author

**Bhargav Mehar**

B.Tech in Computer Science and Engineering | 2026 Graduate

Interests: Data Analytics, Python, SQL, and Machine Learning.
