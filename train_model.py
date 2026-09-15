import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("Iris_dataset.csv")


# ==========================================
# 2. Features and Target
# ==========================================

X = df.drop(columns=["Id", "Species"])
y = df["Species"]


# ==========================================
# 3. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 5. Models
# ==========================================

models = {

    "Logistic Regression":
        LogisticRegression(),

    "KNN":
        KNeighborsClassifier(),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "SVM":
        SVC()
}


# ==========================================
# 6. Train Models
# ==========================================

results = {}
trained_models = {}

print("\n========== MODEL RESULTS ==========\n")

for name, model in models.items():

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy
    trained_models[name] = model

    print(f"{name}: {accuracy:.4f}")


# ==========================================
# 7. Best Model
# ==========================================

best_model_name = max(results, key=results.get)

best_model = trained_models[best_model_name]

print("\n========== BEST MODEL ==========")

print("Best Model:", best_model_name)
print("Best Accuracy:", f"{results[best_model_name]:.4f}")


# ==========================================
# 8. Best Model Prediction
# ==========================================

y_pred_best = best_model.predict(X_test_scaled)


# ==========================================
# 9. Classification Report
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred_best
    )
)


# ==========================================
# 10. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred_best
)

print("\n========== CONFUSION MATRIX ==========")

print(cm)


# ==========================================
# 11. Confusion Matrix Visualization
# ==========================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=best_model.classes_,
    yticklabels=best_model.classes_
)

plt.title("SVM Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.show()


# ==========================================
# 12. Model Comparison Graph
# ==========================================

plt.figure(figsize=(9, 5))

plt.bar(
    results.keys(),
    results.values()
)

plt.title("Model Accuracy Comparison")

plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")

plt.xticks(rotation=20)

plt.ylim(0, 1)

plt.show()

# ==========================================
# 13. Save Best Model and Scaler
# ==========================================

import joblib

joblib.dump(best_model, "iris_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n========== MODEL SAVED ==========")

print("Model saved as: iris_model.pkl")
print("Scaler saved as: scaler.pkl")