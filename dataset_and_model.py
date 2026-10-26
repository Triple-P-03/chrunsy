import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
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

# 6. Initialize models
models = {
    "Support Vector Machine": (
        SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
        True,  # requires scaled features
    ),
    "Decision Tree": (
        DecisionTreeClassifier(max_depth=5, random_state=42),
        False, # tree models don't require scaling, but scale won't harm
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        False,
    ),
}

# 7. Train and evaluate each model
results = {}

for name, (model, use_scaling) in models.items():
    X_tr = X_train_scaled if use_scaling else X_train
    X_te = X_test_scaled if use_scaling else X_test
    
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    weighted_f1 = f1_score(y_test, preds, average="weighted")
    
    results[name] = {
        "model": model,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "predictions": preds,
        "scaled": use_scaling
    }
    
    print(f"\n================ {name.upper()} ================")
    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, preds))
    print("\n--- Classification Report ---")
    print(classification_report(y_test, preds))

# 8. Compare Models
print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)
summary_df = pd.DataFrame([
    {
        "Model": name,
        "Accuracy": f"{metrics['accuracy']:.4f}",
        "Macro F1": f"{metrics['macro_f1']:.4f}",
        "Weighted F1": f"{metrics['weighted_f1']:.4f}",
    }
    for name, metrics in results.items()
])
print(summary_df.to_string(index=False))

# Pick the best model based on Weighted F1-Score (balances precision/recall across imbalanced classes)
best_model_name = max(results, key=lambda k: results[k]["weighted_f1"])
best_model = results[best_model_name]["model"]

print(f"\n---> Best Performing Model: {best_model_name}")

# 9. Save scaler and the best performing model
joblib.dump(scaler, "scaler.pkl")
joblib.dump(best_model, "best_churn_model.pkl")

print(f"Success: 'scaler.pkl' and 'best_churn_model.pkl' ({best_model_name}) saved successfully!")
