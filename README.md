# 🚨 AI-Based Road Accident Severity Prediction and Emergency Alert System

### Machine Learning | Python | Random Forest | Streamlit | Data Analytics

An interactive Machine Learning-based system that predicts road accident severity using historical accident data and demonstrates an automated emergency response escalation protocol based on predicted severity and accident risk factors.

The project combines predictive analytics, data visualization and an interactive dashboard to explore how machine learning can support road safety analysis and emergency response planning.

---

## 📌 Project Overview

Road accidents can result in different levels of injuries depending on several factors, including driving conditions, weather, lighting, collision type, driver experience and the number of casualties.

This project uses a supervised Machine Learning approach to classify accident severity into three categories:

- **Slight Injury**
- **Serious Injury**
- **Fatal Injury**

A Random Forest Classifier is trained on historical road traffic accident records. The trained model is integrated into a Streamlit dashboard where users can enter incident parameters and obtain a predicted severity level.

The application also implements a rule-based emergency escalation mechanism that displays recommended response actions according to the predicted severity and calculated crash risk index.

## ✨ Key Features

### 1. Accident Severity Prediction
- Predicts accident severity using a trained Random Forest Classifier.
- Supports three severity categories: Slight, Serious and Fatal Injury.
- Uses 12 selected accident-related input features.
- Loads previously trained model artifacts for prediction.

### 2. Emergency Alert and Escalation Protocol
- Calculates a composite crash risk index.
- Combines ML predictions with additional rule-based risk conditions.
- Classifies incidents into three emergency response levels:
  - Level 1: Critical / Fatal Risk
  - Level 2: Serious Injury Risk
  - Level 3: Slight / Minor Injury
- Displays recommended emergency response actions.
- Generates a dispatch reference ID and records incident details in a session-based dispatch log.

### 3. Dataset Analytics and Visualization

The dashboard provides analytical visualizations, including:

- Distribution of Road Accident Severity.
- Hourly Accident Frequency Analysis.
- Random Forest Feature Importance.
- Model Confusion Matrix.

### 4. Interactive Dashboard
- User-friendly Streamlit interface.
- Incident location and environmental parameter inputs.
- Interactive prediction results.
- Dataset overview and sample raw records.
- Emergency alert dispatch log.

---

## 📊 Dataset Information

**Dataset:** Road Traffic Accident Dataset of Addis Ababa City

**Source:** Mendeley Data

**Author:** Tarikwa Tesfa Bedane

**DOI:** 10.17632/xytv86278f.1

**Dataset Size:** 12,316 records

**Attributes:** 32 columns

**Data Period:** 2017–2020

The dataset is based on manual road accident records collected from Addis Ababa sub-city police departments.

Dataset distribution:

| Accident Severity | Records |
|---|---:|
| Slight Injury | 10,415 |
| Serious Injury | 1,743 |
| Fatal Injury | 158 |
| **Total** | **12,316** |

Dataset source: https://data.mendeley.com/datasets/xytv86278f/1

### Selected Model Features

The model uses the following 12 input features:

1. Hour of Accident
2. Day of Week
3. Driver Age Group
4. Driving Experience
5. Accident Area
6. Road Surface Conditions
7. Lighting Conditions
8. Weather Conditions
9. Type of Collision
10. Number of Vehicles Involved
11. Number of Casualties
12. Cause of Accident

The target variable is `Accident_severity`.

---

## ⚙️ Machine Learning Methodology

### Algorithm: Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to perform classification.

The project uses the following configuration:

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Number of Estimators | 150 |
| Maximum Depth | 16 |
| Class Weight | Balanced |
| Random State | 42 |
| Training Split | 80% |
| Testing Split | 20% |

### Data Preprocessing

The preprocessing workflow includes:

- Extracting the hour from the accident time.
- Handling missing categorical values using mode imputation.
- Handling missing numerical values using median imputation.
- Converting categorical variables into numerical representations using Label Encoding.
- Separating input features and target labels.
- Performing a stratified train-test split.

### Model Evaluation

The trained model is evaluated using:

- Accuracy Score
- Classification Report
- Confusion Matrix
- Feature Importance Analysis

**Reproduced Test Accuracy: Approximately 79.14%**

The dataset contains a significant imbalance between severity classes. Consequently, overall accuracy should not be interpreted as equal predictive reliability across all three categories. In particular, fatal-injury classification requires further improvement.

---

## 🚑 Emergency Escalation Logic

The application combines predicted severity with a rule-based crash risk index.

The risk index is calculated using:

`Risk Score = (Casualties × 18) + (Vehicles × 8) + Additional Risk Factors`

Additional risk factors:
- Darkness conditions: +15
- Wet road surface or rainy weather: +12

The displayed risk score is capped at 99 out of 100.

### Emergency Levels

| Level | Condition | Response Demonstration |
|---|---|---|
| Level 1 | Fatal prediction or risk score ≥ 75 | Advanced life support, trauma hospital and rescue escalation |
| Level 2 | Serious prediction or risk score ≥ 48 | Basic life support ambulance and police traffic control notification |
| Level 3 | Other predictions | First-aid assistance and traffic management |

The application also generates a dispatch reference ID and maintains a session-based incident log.

**Note:** Emergency actions are simulated within the dashboard. The current prototype does not actually contact emergency services, hospitals or emergency contacts.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Dataset manipulation and preprocessing |
| NumPy | Numerical computations |
| Scikit-learn | Machine Learning and model evaluation |
| Random Forest | Accident severity classification |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive web dashboard |
| Joblib | Saving and loading trained model artifacts |

---

## 📁 Project Structure

```text
Road_Accident_Project/
│
├── app.py
├── train_model.py
├── RTA Dataset.csv
├── requirements.txt
├── README.md
│
├── accident_model.pkl
├── encoders.pkl
├── features.pkl
│
├── plots/
│   ├── severity_distribution.png
│   ├── hourly_accidents.png
│   ├── feature_importance.png
│   └── confusion_matrix.png
│
└── Project Screenshots/
```

The model artifacts and visualization files are generated by the training script.

---

## 🚀 Installation and Execution

### Prerequisites

- Python 3.10 or above
- pip package manager

### Step 1: Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Navigate into the project directory:

```bash
cd Road_Accident_Project
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train the Machine Learning Model

Run:

```bash
python train_model.py
```

This script:

- Loads the accident dataset.
- Performs data preprocessing.
- Generates analytical visualizations.
- Trains the Random Forest Classifier.
- Evaluates model performance.
- Saves the trained model, encoders and feature list.

### Step 4: Launch the Dashboard

Run:

```bash
python -m streamlit run app.py
```

The application will open in your default browser.

Local URL:

`http://localhost:8501`

---

## 📈 Dashboard Modules

**1. Live Prediction & Emergency Dispatch**

Enter incident information and obtain a predicted accident severity level with a corresponding emergency escalation recommendation.

**2. Dataset Analytics & Visuals**

Explore accident severity distribution, hourly accident frequency, feature importance and the model confusion matrix.

**3. Alert Dispatch Log**

View incident records generated during the current application session.

---

## 🔮 Future Enhancements

- Integration with verified real-time traffic and accident data sources.
- GPS-based incident location detection and mapping.
- Integration with emergency service APIs.
- SMS and notification-based emergency alerts.
- Improved performance on minority severity classes.
- Advanced model comparison using XGBoost, Decision Trees and other classifiers.
- Deployment on a cloud platform.
- Persistent incident database and historical dispatch monitoring.

---

## ⚠️ Disclaimer

This project is developed for educational and research purposes. Predictions are based on historical accident records and should not be treated as guaranteed assessments of actual accident severity.

The emergency escalation module demonstrates rule-based software functionality and is not connected to real-world emergency response services.

---

## 👨‍💻 Author

**Kartikey Kamboj**

B.Tech – Computer Science and Engineering

Interested in Machine Learning, Data Analytics and Software Development.

