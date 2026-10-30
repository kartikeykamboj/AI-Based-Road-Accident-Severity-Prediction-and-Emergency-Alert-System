import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
os.makedirs("plots", exist_ok=True)

DATASET_FILE = "RTA Dataset.csv"

print("Loading dataset...")
df = pd.read_csv(DATASET_FILE)
print(f"Dataset shape: {df.shape}")

# Extract Hour from Time column
df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.hour
df["Hour"] = df["Hour"].fillna(df["Hour"].median()).astype(int)

selected_features = [
    "Hour",
    "Day_of_week",
    "Age_band_of_driver",
    "Driving_experience",
    "Area_accident_occured",
    "Road_surface_conditions",
    "Light_conditions",
    "Weather_conditions",
    "Type_of_collision",
    "Number_of_vehicles_involved",
    "Number_of_casualties",
    "Cause_of_accident"
]

target_col = "Accident_severity"

categorical_cols = [
    "Day_of_week",
    "Age_band_of_driver",
    "Driving_experience",
    "Area_accident_occured",
    "Road_surface_conditions",
    "Light_conditions",
    "Weather_conditions",
    "Type_of_collision",
    "Cause_of_accident"
]

numeric_cols = ["Hour", "Number_of_vehicles_involved", "Number_of_casualties"]

# Keep required columns and handle missing values cleanly
data = df[selected_features + [target_col]].copy()
for col in categorical_cols + [target_col]:
    data[col] = data[col].fillna(data[col].mode()[0]).astype(str)
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce").fillna(data[col].median()).astype(int)

print("Generating EDA visualizations...")

# Plot 1: Severity Distribution
plt.figure(figsize=(7, 4.5))
ax = sns.countplot(
    data=data,
    x=target_col,
    hue=target_col,
    palette=["#2ecc71", "#f39c12", "#e74c3c"],
    order=["Slight Injury", "Serious Injury", "Fatal injury"],
    legend=False
)
plt.title("Distribution of Road Accident Severity", fontsize=13, fontweight="bold")
plt.xlabel("Accident Severity Level")
plt.ylabel("Total Recorded Incidents")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/severity_distribution.png", dpi=300)
plt.close()

# Plot 2: Accidents by Hour of Day
plt.figure(figsize=(9, 4.5))
sns.histplot(data=data, x="Hour", bins=24, kde=True, color="#3498db")
plt.title("Hourly Accident Frequency Analysis (24-Hour Cycle)", fontsize=13, fontweight="bold")
plt.xlabel("Hour of the Day (0 - 23 hrs)")
plt.ylabel("Number of Accidents")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig("plots/hourly_accidents.png", dpi=300)
plt.close()

# Encode categorical variables explicitly (compatible with Pandas 2.x and 3.x)
encoders = {}
encoded_data = data.copy()

for col in categorical_cols:
    le = LabelEncoder()
    encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
    encoders[col] = le

target_le = LabelEncoder()
encoded_data[target_col] = target_le.fit_transform(encoded_data[target_col].astype(str))
encoders[target_col] = target_le

X = encoded_data[selected_features]
y = encoded_data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=16,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Training Completed! Test Accuracy: {acc * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_le.classes_))

# Plot 3: Feature Importance
plt.figure(figsize=(8, 5))
importances = pd.Series(model.feature_importances_, index=selected_features).sort_values(ascending=True)
importances.plot(kind="barh", color="#2c3e50")
plt.title("Key Factors Influencing Accident Severity (Random Forest)", fontsize=13, fontweight="bold")
plt.xlabel("Relative Importance Score")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=300)
plt.close()

# Plot 4: Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=target_le.classes_,
    yticklabels=target_le.classes_
)
plt.title("Model Confusion Matrix", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Severity")
plt.ylabel("Actual Severity")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=300)
plt.close()

# Save artifacts for dashboard
joblib.dump(model, "accident_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(selected_features, "features.pkl")
print("Saved model, encoders, and plots inside 'plots/' folder.")