Iris Flower Classification

An end-to-end Machine Learning project that classifies Iris flowers into three species based on their sepal and petal measurements. The project covers the complete ML workflow — from data exploration and preprocessing to model training, evaluation, model saving, and deployment through a Flask web application.

🚀 Project Overview

The objective of this project is to build a reliable classification system that predicts the species of an Iris flower using four numerical features:

Sepal Length
Sepal Width
Petal Length
Petal Width

Multiple Machine Learning classification algorithms were trained and evaluated. Among them, Support Vector Machine (SVM) achieved the highest test accuracy of 96.67% and was selected as the final model.

The trained model was integrated into a Flask-based web application that allows users to enter flower measurements and receive a real-time prediction.

🎯 Objectives
Explore and understand the Iris dataset
Perform Exploratory Data Analysis (EDA)
Clean and preprocess the data
Split the dataset into training and testing sets
Apply feature scaling using StandardScaler
Train multiple classification models
Compare model performance
Select the best-performing model
Save the trained model and scaler
Build a Flask web application for real-time prediction
📊 Dataset

The project uses the classic Iris Flower Dataset containing 150 samples and 3 species.

Species
Iris-setosa
Iris-versicolor
Iris-virginica

Each species contains 50 samples.

Features
Feature	Description
SepalLengthCm	Sepal length in centimeters
SepalWidthCm	Sepal width in centimeters
PetalLengthCm	Petal length in centimeters
PetalWidthCm	Petal width in centimeters

The Id column was excluded from model training because it does not represent a meaningful predictive feature.

🔍 Exploratory Data Analysis

The project includes several EDA steps to understand the dataset:

Dataset structure and dimensions
Statistical summary
Species distribution
Missing-value analysis
Duplicate-value analysis
Feature distributions
Feature relationships
Correlation analysis
Dataset Summary
Total Samples: 150
Total Features: 4 predictive features
Missing Values: 0
Duplicate Rows: 0
Classes: 3
Samples per Class: 50
⚙️ Machine Learning Pipeline

The complete workflow follows:

Iris Dataset
      ↓
Data Loading
      ↓
Exploratory Data Analysis
      ↓
Data Preprocessing
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Best Model Selection
      ↓
Model Saving
      ↓
Flask Web Application
      ↓
Real-Time Prediction
Train-Test Split

The dataset was divided into:

80% Training Data → 120 samples
20% Testing Data → 30 samples

A stratified split was used to preserve the class distribution.

Feature Scaling

StandardScaler was used to standardize the numerical features before training the models.

🤖 Models Trained

The following classification algorithms were evaluated:

Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree
Random Forest
Support Vector Machine (SVM)
Model Performance
Model	Test Accuracy
Logistic Regression	93.33%
K-Nearest Neighbors	93.33%
Decision Tree	90.00%
Random Forest	90.00%
Support Vector Machine	96.67%
🏆 Final Model

Support Vector Machine (SVM) was selected as the final model based on its test performance.

Test Accuracy: 96.67%

The model correctly classified 29 out of 30 test samples.

📈 Evaluation

The final SVM model was evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
Confusion Matrix
[[10, 0, 0],
 [0, 9, 1],
 [0, 0, 10]]

The model correctly classified all Iris-setosa and Iris-virginica samples in the test set. One Iris-versicolor sample was classified as Iris-virginica.

🌐 Flask Web Application

The trained SVM model was integrated into a Flask web application.

Users can enter:

Sepal Length
Sepal Width
Petal Length
Petal Width

The application processes the input using the saved scaler and returns the predicted Iris species in real time.

Application Features
Clean and responsive interface
User-friendly input form
Real-time prediction
Flask backend
Saved ML model integration
Responsive HTML/CSS frontend
🛠️ Technologies Used
Programming Language
Python
Machine Learning
Scikit-learn
SVM
Logistic Regression
KNN
Decision Tree
Random Forest
Data Analysis
Pandas
NumPy
Data Visualization
Matplotlib
Seaborn
Web Development
Flask
HTML5
CSS3
Model Serialization
Joblib
Development Tools
VS Code
Git
GitHub

📁 Project Structure

Iris-Flower-Classification/
│
├── Iris_dataset.csv
├── Iris.csv
│
├── eda.py
├── train_model.py
├── predict.py
├── app.py
│
├── iris_model.pkl
├── scaler.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── requirements.txt
├── README.md
├── .gitignore
│
└── venv/

venv/ is excluded from version control using .gitignore.


💻 Installation & Setup

1. Clone the Repository
git clone https://github.com/PranoyChandraDas107/Iris-Flower-Classification.git
2. Navigate to the Project Directory
cd Iris-Flower-Classification
3. Create a Virtual Environment
python -m venv venv
4. Activate the Virtual Environment

Windows PowerShell:

venv\Scripts\Activate.ps1

Windows CMD:

venv\Scripts\activate

5. Install Dependencies
pip install -r requirements.txt

▶️ Running the Project

Run Machine Learning Prediction
python predict.py

Enter the four flower measurements when prompted.

Run the Flask Web Application
python app.py

Then open:

http://127.0.0.1:5000

The web application will open in your browser.

🧪 Example Prediction

Example input:

Sepal Length: 5.1
Sepal Width: 3.5
Petal Length: 1.4
Petal Width: 0.2

Prediction:

Iris-setosa

📌 Key Highlights

Complete end-to-end Machine Learning workflow
Exploratory Data Analysis
Data preprocessing and feature scaling
Comparison of 5 classification algorithms
96.67% test accuracy using SVM
Model serialization using Joblib
Flask-based real-time prediction system
Git and GitHub version control

🔮 Future Improvements

Possible future enhancements include:

Deploy the Flask application online
Add prediction probability
Add interactive data visualizations
Improve UI/UX
Add input validation and error handling
Add REST API support
Containerize the application using Docker

👨‍💻 Author

Pranoy Chandra Das

Software Engineering | Data Science | Machine Learning

📄 License

This project is intended for educational and learning purposes.
