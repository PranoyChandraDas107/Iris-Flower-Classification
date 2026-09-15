import pandas as pd

# Load dataset
df = pd.read_csv("Iris_dataset.csv")

print("========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== SPECIES DISTRIBUTION ==========")
print(df["Species"].value_counts())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Species Distribution
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Species")

plt.title("Distribution of Iris Species")
plt.xlabel("Species")
plt.ylabel("Count")

plt.show()


# ==========================================
# 2. Feature Distribution
# ==========================================

df[[
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]].hist(figsize=(10, 8))

plt.suptitle("Feature Distribution")

plt.show()


# ==========================================
# 3. Pairplot
# ==========================================

sns.pairplot(df, hue="Species")

plt.show()


# ==========================================
# 4. Correlation Heatmap
# ==========================================

correlation_df = df.drop(columns=["Species"])

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")

plt.show()