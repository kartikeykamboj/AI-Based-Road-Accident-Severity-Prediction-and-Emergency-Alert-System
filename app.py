import os
import time
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Road Accident Severity & Emergency Alert System",
    layout="wide"
)

# Load trained model and encoders
@st.cache_resource
def load_artifacts():
    model = joblib.load("accident_model.pkl")
    encoders = joblib.load("encoders.pkl")
    features = joblib.load("features.pkl")
    df = pd.read_csv("RTA Dataset.csv")
    return model, encoders, features, df

if not os.path.exists("accident_model.pkl"):
    st.error("Model files not found! Please run `python train_model.py` first.")
    st.stop()

model, encoders, features, df = load_artifacts()

st.title("AI-Based Road Accident Severity Prediction & Emergency Alert System")
st.markdown("Real-time crash severity assessment, data analytics, and automated emergency response dispatch.")

tab1, tab2, tab3 = st.tabs([
    "Live Prediction & Emergency Dispatch", 
    "Dataset Analytics & Visuals", 
    "Alert Dispatch Log"
])

if "dispatch_logs" not in st.session_state:
    st.session_state.dispatch_logs = []

with tab1:
    col_input, col_result = st.columns([1.2, 1])
    
    with col_input:
        st.subheader("Incident Site Parameters")
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            
            with c1:
                location_zone = st.text_input("GPS Coordinates / Highway Sector", "NH-24 Sector 62 Overpass (28.6208° N, 77.3639° E)")
                hour = st.slider("Hour of Incident (24-Hr Format)", 0, 23, 21)
                day = st.selectbox("Day of Week", encoders["Day_of_week"].classes_)
                age_band = st.selectbox("Driver Age Group", encoders["Age_band_of_driver"].classes_)
                experience = st.selectbox("Driving Experience", encoders["Driving_experience"].classes_)
                area = st.selectbox("Area Type", encoders["Area_accident_occured"].classes_)
                
            with c2:
                road_cond = st.selectbox("Road Surface Condition", encoders["Road_surface_conditions"].classes_)
                light_cond = st.selectbox("Lighting Condition", encoders["Light_conditions"].classes_)
                weather = st.selectbox("Weather Condition", encoders["Weather_conditions"].classes_)
                collision = st.selectbox("Collision Type", encoders["Type_of_collision"].classes_)
                vehicles = st.number_input("Number of Vehicles Involved", min_value=1, max_value=15, value=2)
                casualties = st.number_input("Estimated Number of Casualties", min_value=1, max_value=25, value=2)
                
            cause = st.selectbox("Primary Reported Cause", encoders["Cause_of_accident"].classes_)
            submit_btn = st.form_submit_button("Predict Severity & Trigger Emergency Protocol", use_container_width=True)

    with col_result:
        st.subheader("AI Assessment & Emergency Protocol")
        if submit_btn:
            input_dict = {
                "Hour": hour,
                "Day_of_week": encoders["Day_of_week"].transform([day])[0],
                "Age_band_of_driver": encoders["Age_band_of_driver"].transform([age_band])[0],
                "Driving_experience": encoders["Driving_experience"].transform([experience])[0],
                "Area_accident_occured": encoders["Area_accident_occured"].transform([area])[0],
                "Road_surface_conditions": encoders["Road_surface_conditions"].transform([road_cond])[0],
                "Light_conditions": encoders["Light_conditions"].transform([light_cond])[0],
                "Weather_conditions": encoders["Weather_conditions"].transform([weather])[0],
                "Type_of_collision": encoders["Type_of_collision"].transform([collision])[0],
                "Number_of_vehicles_involved": vehicles,
                "Number_of_casualties": casualties,
                "Cause_of_accident": encoders["Cause_of_accident"].transform([cause])[0]
            }
            
            input_df = pd.DataFrame([input_dict])
            pred_encoded = model.predict(input_df)[0]
            pred_probs = model.predict_proba(input_df)[0]
            pred_label = encoders["Accident_severity"].inverse_transform([pred_encoded])[0]
            
            # Emergency Escalation Logic for Multi-Casualty / High-Risk Night Crashes
            risk_score = (casualties * 18) + (vehicles * 8)
            if "Darkness" in light_cond:
                risk_score += 15
            if "Wet" in road_cond or "Rain" in weather:
                risk_score += 12
                
            if pred_label == "Fatal injury" or risk_score >= 75:
                final_severity = "CRITICAL / FATAL RISK (LEVEL 1)"
                alert_color = "error"
                protocol = [
                    "ALS (Advanced Life Support) Trauma Ambulance Dispatched",
                    "Nearest Level-1 Trauma Hospital ICU Alerted",
                    "Highway Patrol & Fire Rescue Unit Mobilized",
                    "Automated SOS SMS Broadcast Sent to Emergency Contacts"
                ]
            elif pred_label == "Serious Injury" or risk_score >= 48:
                final_severity = "SERIOUS INJURY RISK (LEVEL 2)"
                alert_color = "warning"
                protocol = [
                    "BLS (Basic Life Support) Ambulance Dispatched",
                    "Local Police Traffic Control Unit Notified",
                    "Nearest District Hospital ER Placed on Standby"
                ]
            else:
                final_severity = "SLIGHT / MINOR INJURY (LEVEL 3)"
                alert_color = "success"
                protocol = [
                    "First-Aid Patrol Vehicle Dispatched",
                    "Traffic Clearing & Tow Assistance Assigned",
                    "Incident Logged in Municipal Road Safety Database"
                ]

            if alert_color == "error":
                st.error(f"PREDICTED SEVERITY: {final_severity}")
            elif alert_color == "warning":
                st.warning(f"PREDICTED SEVERITY: {final_severity}")
            else:
                st.success(f"PREDICTED SEVERITY: {final_severity}")

            st.metric("Composite Crash Risk Index", f"{min(risk_score, 99)} / 100")
            
            st.markdown("#### Automated Emergency Dispatch Actions:")
            for item in protocol:
                st.write(f"- {item}")
                
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.info(f"Dispatch Reference ID: #ERS-{np.random.randint(10000, 99999)} | Logged at: {timestamp}\nLocation: {location_zone}")
            
            st.session_state.dispatch_logs.insert(0, {
                "Timestamp": timestamp,
                "Location": location_zone,
                "Predicted Severity": final_severity,
                "Vehicles": vehicles,
                "Casualties": casualties,
                "Weather": weather,
                "Lighting": light_cond
            })
        else:
            st.info("Configure incident details on the left and click 'Predict Severity & Trigger Emergency Protocol' to run real-time classification.")

with tab2:
    st.subheader("Dataset Overview & Model Performance Visuals")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Official Records", f"{len(df):,}")
    m2.metric("Total Features Analyzed", f"{df.shape[1]}")
    m3.metric("Classifier Algorithm", "Random Forest")
    m4.metric("Validation Split", "80% Train / 20% Test")
    
    st.markdown("---")
    g1, g2 = st.columns(2)
    with g1:
        if os.path.exists("plots/severity_distribution.png"):
            st.image("plots/severity_distribution.png", caption="Class Distribution in Dataset")
        if os.path.exists("plots/feature_importance.png"):
            st.image("plots/feature_importance.png", caption="Random Forest Feature Importance")
    with g2:
        if os.path.exists("plots/hourly_accidents.png"):
            st.image("plots/hourly_accidents.png", caption="24-Hour Accident Trend")
        if os.path.exists("plots/confusion_matrix.png"):
            st.image("plots/confusion_matrix.png", caption="Confusion Matrix on Test Set")
            
    st.markdown("### Sample Raw Dataset Records")
    st.dataframe(df.head(15), use_container_width=True)

with tab3:
    st.subheader("Recent Emergency Alert Dispatch Logs")
    if st.session_state.dispatch_logs:
        st.dataframe(pd.DataFrame(st.session_state.dispatch_logs), use_container_width=True)
    else:
        st.write("No emergency alerts triggered in this session yet.")