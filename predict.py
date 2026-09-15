import pandas as pd
import joblib


# ==========================================
# Load Saved Model and Scaler
# ==========================================

model = joblib.load("iris_model.pkl")
scaler = joblib.load("scaler.pkl")


# ==========================================
# Take User Input
# ==========================================

print("===== Iris Flower Prediction =====")

sepal_length = float(input("Enter Sepal Length (cm): "))
sepal_width = float(input("Enter Sepal Width (cm): "))
petal_length = float(input("Enter Petal Length (cm): "))
petal_width = float(input("Enter Petal Width (cm): "))


# ==========================================
# Create Input Array
# ==========================================

input_data = pd.DataFrame([
    {
        "SepalLengthCm": sepal_length,
        "SepalWidthCm": sepal_width,
        "PetalLengthCm": petal_length,
        "PetalWidthCm": petal_width
    }
])


# Scale Input

input_scaled = scaler.transform(input_data)


# Prediction

prediction = model.predict(input_scaled)


print("\n========== PREDICTION ==========")
print("Predicted Species:", prediction[0])