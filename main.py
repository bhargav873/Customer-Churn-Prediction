# Import Libraries

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Loading Dataset

df = pd.read_csv(r"C:\Users\justi\OneDrive\Desktop\Customer Churn Prediction\Telco-Customer-Churn.csv")

print("Dataset loaded successfully!")
print(df.head())


# Exploring the Dataset

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe(include="all"))


# Data The Cleaning

print("\n========== DATA CLEANING ==========")

df = df.drop_duplicates()

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDataset shape after cleaning:")
print(df.shape)

print("\nFirst five cleaned rows:")
print(df.head())


# Customer Churn Analysis

print("\n========== CUSTOMER CHURN ANALYSIS ==========")

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

df["Churn"].value_counts().plot(
    kind="bar",
    title="Customer Churn Distribution"
)

plt.xlabel("Churn (No = Stayed, Yes = Left)")
plt.ylabel("Number of Customers")
plt.tight_layout()


# Preparing Data for Machine Learning

print("\n========== DATA PREPARATION ==========")

# Remove customer ID
df = df.drop(columns=["customerID"])

# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Convert categorical data into numerical data
X = pd.get_dummies(X, drop_first=True)

# Fill missing values
X = X.fillna(X.median())

# Scale the features
scaler = StandardScaler()

X = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# Display prepared data
print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

print("\nData preparation completed successfully!")


# Splitting The Dataset

print("\n========== TRAIN-TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# Training the Machine Learning Model

print("\n========== MODEL TRAINING ==========")

model = LogisticRegression(max_iter=3000)

model.fit(X_train, y_train)

print("Model training completed successfully!")


# Make Predictions and Check Accuracy

print("\n========== MODEL EVALUATION ==========")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")


# Classification Report

print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Stayed", "Churned"],
    zero_division=0
))

df["Churn"].value_counts().plot(
    kind="bar",
    title="Customer Churn Distribution"
)

plt.xlabel("Churn (No = Stayed, Yes = Left)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# Confusion Matrix

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Stayed", "Churned"]
)

plt.title("Customer Churn Confusion Matrix")
plt.show()


# Confusion Matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("\n========== CONFUSION MATRIX ==========")

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

# Display the confusion matrix
cm_display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stayed", "Churned"]
)

cm_display.plot()
plt.title("Customer Churn Prediction - Confusion Matrix")
plt.tight_layout()
plt.show()

# Module 11: Feature Importance

print("\n========== FEATURE IMPORTANCE ==========")

# Get feature names and model coefficients
feature_names = X.columns
coefficients = model.coef_[0]

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

# Sort by absolute coefficient value
importance_df["Importance"] = importance_df["Coefficient"].abs()
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Display the top 10 features
print("\nTop 10 Important Features:")
print(importance_df[["Feature", "Coefficient"]].head(10))

# Plotting the top 10 features by using Matplotlib
top_features = importance_df.head(10).sort_values(
    by="Coefficient"
)

top_features.plot(
    x="Feature",
    y="Coefficient",
    kind="barh",
    legend=False,
    title="Top 10 Features Associated with Churn Predictions"
)

plt.xlabel("Model Coefficient")
plt.ylabel("Customer Features")
plt.tight_layout()
plt.show()


# Comparision of Machine Learning Algorithms

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("\n========== MODEL COMPARISON ==========")

# Prepare features and target
# customerID was already removed in Module 5
X_comparison = df.drop(columns=["Churn"], errors="ignore").copy()
y_comparison = df["Churn"].map({"No": 0, "Yes": 1})

# Convert categorical columns into numerical columns
X_comparison = pd.get_dummies(X_comparison, drop_first=True)

# Split the data before fitting preprocessing
X_train_compare, X_test_compare, y_train_compare, y_test_compare = train_test_split(
    X_comparison,
    y_comparison,
    test_size=0.2,
    random_state=42,
    stratify=y_comparison
)

# Define the models
models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000))
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])
}

results = []

# Train and evaluate each model
for model_name, model in models.items():

    model.fit(X_train_compare, y_train_compare)

    predictions = model.predict(X_test_compare)

    results.append({
        "Model": model_name,
        "Accuracy": round(
            accuracy_score(y_test_compare, predictions), 3
        ),
        "Precision": round(
            precision_score(y_test_compare, predictions, zero_division=0), 3
        ),
        "Recall": round(
            recall_score(y_test_compare, predictions, zero_division=0), 3
        ),
        "F1 Score": round(
            f1_score(y_test_compare, predictions, zero_division=0), 3
        )
    })

# Display the comparison
results_df = pd.DataFrame(results)

print("\nModel Performance Comparison:")
print(results_df.to_string(index=False))

print("\nModel comparison completed successfully!")



# Improving the Churn Detection

from sklearn.metrics import confusion_matrix

print("\n========== IMPROVING CHURN DETECTION ==========")

# Create a Random Forest model
improved_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# Train the model
improved_model.fit(X_train_compare, y_train_compare)

# Get churn probabilities
churn_probabilities = improved_model.predict_proba(X_test_compare)[:, 1]

# Test different prediction thresholds
thresholds = [0.3, 0.4, 0.5]

print("\nPerformance at Different Thresholds:")

for threshold in thresholds:

    predictions = (churn_probabilities >= threshold).astype(int)

    recall = recall_score(
        y_test_compare,
        predictions,
        zero_division=0
    )

    precision = precision_score(
        y_test_compare,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test_compare,
        predictions,
        zero_division=0
    )

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_compare, predictions))

print("\nThreshold comparison completed successfully!")


# Module 14: Business Insights and Visualizations

import matplotlib.pyplot as plt

print("\n========== BUSINESS INSIGHTS AND VISUALIZATIONS ==========")

# Fiirstly Create a separate dataset for business analysis
business_df = df.copy()

# 1. Overall churn percentage
churn_counts = business_df["Churn"].value_counts()

plt.figure(figsize=(6, 4))
churn_counts.plot(kind="bar")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 2. Churn rate by contract type
contract_churn = pd.crosstab(
    business_df["Contract"],
    business_df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Contract Type (%):")
print(contract_churn.round(2))

contract_churn.plot(
    kind="bar",
    figsize=(8, 5)
)
plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 3. Churn rate by tenure group
business_df["TenureGroup"] = pd.cut(
    business_df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months"]
)

tenure_churn = pd.crosstab(
    business_df["TenureGroup"],
    business_df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Tenure Group (%):")
print(tenure_churn.round(2))

tenure_churn.plot(
    kind="bar",
    figsize=(8, 5)
)
plt.title("Customer Churn by Tenure")
plt.xlabel("Customer Tenure")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 4. Churn rate by internet service
internet_churn = pd.crosstab(
    business_df["InternetService"],
    business_df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Internet Service (%):")
print(internet_churn.round(2))

internet_churn.plot(
    kind="bar",
    figsize=(8, 5)
)
plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nBusiness insights and visualizations completed successfully!")


# Prediction Churn for an Individual Customer

print("\n========== INDIVIDUAL CUSTOMER CHURN PREDICTION ==========")

# Create a sample customer using the original dataset's columns
customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 6,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 90.0,
    "TotalCharges": 540.0
}

# Convert customer details into a DataFrame
customer_df = pd.DataFrame([customer])

# Match the feature columns used during training
customer_df = pd.get_dummies(customer_df)

customer_df = customer_df.reindex(
    columns=X_train_compare.columns,
    fill_value=0
)

# Predict churn probability
churn_probability = improved_model.predict_proba(customer_df)[0][1]

# Use the selected threshold
threshold = 0.4

if churn_probability >= threshold:
    prediction = "Customer is likely to churn"
else:
    prediction = "Customer is likely to stay"

print("\nCustomer Details:")
print("Tenure:", customer["tenure"], "months")
print("Contract:", customer["Contract"])
print("Internet Service:", customer["InternetService"])
print("Monthly Charges: ₹", customer["MonthlyCharges"])

print("\nPredicted Churn Probability:",
      round(churn_probability * 100, 2), "%")

print("Prediction:", prediction)

print("\nIndividual customer prediction completed successfully!")

id="w5k8zc"
# Module 16: Save the Trained Model

import joblib
import os

print("\n========== SAVING THE TRAINED MODEL ==========")

# Create a folder to store the trained model
model_folder = "saved_model"
os.makedirs(model_folder, exist_ok=True)

# Save the trained Random Forest pipeline
model_path = os.path.join(
    model_folder,
    "customer_churn_model.pkl"
)

joblib.dump(improved_model, model_path)

print("\nModel saved successfully!")
print("Model location:", os.path.abspath(model_path))

print("\nModel saving completed successfully!")











































