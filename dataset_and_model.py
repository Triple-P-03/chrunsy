import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Download dataset directly from source
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# 2. Clean TotalCharges (convert empty spaces to 0)
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True)
)
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# 3. Clean and map target and categorical features
df["Churn"] = df["Churn"].astype(str).str.strip()
df["Churn_Numeric"] = df["Churn"].map({"Yes": 1, "No": 0})
df["Contract_Numeric"] = df["Contract"].apply(
    lambda x: 0 if "Month-to-month" in str(x) else 1
)

# 4. Select features and split dataset
features = ["tenure", "MonthlyCharges", "TotalCharges", "Contract_Numeric"]
X = df[features]
y = df["Churn_Numeric"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Fit StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Support Vector Machine Classifier
svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# 7. Print Model Performance Metrics
y_pred = svm_model.predict(X_test_scaled)
print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# 8. Save model and scaler files to disk
joblib.dump(svm_model, "svm_churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nSuccess: 'svm_churn_model.pkl' and 'scaler.pkl' generated successfully!")
