
from flask import Flask, render_template, request
import pandas as pd
import joblib


# ==========================================
# Create Flask Application
# ==========================================

app = Flask(__name__)


# ==========================================
# Load Trained Model and Scaler
# ==========================================

model = joblib.load("iris_model.pkl")
scaler = joblib.load("scaler.pkl")


# ==========================================
# Home Route
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# Prediction Route
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # Get values from HTML form
    sepal_length = float(request.form["sepal_length"])
    sepal_width = float(request.form["sepal_width"])
    petal_length = float(request.form["petal_length"])
    petal_width = float(request.form["petal_width"])


    # Create DataFrame
    input_data = pd.DataFrame([
        {
            "SepalLengthCm": sepal_length,
            "SepalWidthCm": sepal_width,
            "PetalLengthCm": petal_length,
            "PetalWidthCm": petal_width
        }
    ])


    # Scale input
    input_scaled = scaler.transform(input_data)


    # Make prediction
    prediction = model.predict(input_scaled)


    # Send prediction to HTML
    return render_template(
        "index.html",
        prediction=prediction[0]
    )


# ==========================================
# Run Flask Application
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)
