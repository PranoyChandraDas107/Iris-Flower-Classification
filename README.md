# Iris Flower Classification 

A Machine Learning project that predicts the species of an Iris flower based on its sepal and petal measurements.

##  Project Overview

This project uses Machine Learning classification algorithms to predict one of the following Iris species:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

The project includes data analysis, preprocessing, model training, evaluation, and a Flask-based web application for real-time prediction.

##  Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Flask
- Joblib
- HTML
- CSS

##  Dataset

The project uses the Iris dataset containing 150 samples and 4 main features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The dataset contains three classes:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

Each class contains 50 samples.

## Machine Learning Models

The following classification algorithms were trained and evaluated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. Support Vector Machine (SVM)

### Best Model

Support Vector Machine (SVM) achieved the highest test accuracy:

**96.67%**

## 📈 Model Performance

| Model | Accuracy |
|---|---:|
| Logistic Regression | 93.33% |
| KNN | 93.33% |
| Decision Tree | 90.00% |
| Random Forest | 90.00% |
| SVM | 96.67% |

## 🌐 Web Application

The project includes a Flask web application where users can enter flower measurements and get the predicted Iris species.

### Input Features

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

## 📁 Project Structure

```text
Iris-Flower-Classification/
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── Iris_dataset.csv
├── eda.py
├── train_model.py
├── predict.py
├── app.py
├── iris_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── .gitignore