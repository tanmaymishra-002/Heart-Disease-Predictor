import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and column order
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

st.title("Heart Disease Risk Predictor")
st.write("Enter patient details to estimate heart disease risk.")
st.warning("This is a learning project, not a medical diagnostic tool.")

# --- User inputs ---
age = st.slider("Age", 20, 90, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
chol = st.slider("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])

cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
restecg = st.selectbox("Resting ECG Result", [0, 1, 2])
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
thal = st.selectbox("Thal Result", [1, 2, 3])

if st.button("Predict"):
    input_dict = {
        'age': age,
        'sex': 1 if sex == "Male" else 0,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': 1 if fbs == "Yes" else 0,
        'thalach': thalach,
        'exang': 1 if exang == "Yes" else 0,
        'oldpeak': oldpeak,
        'ca': ca,
        'cp': cp,
        'restecg': restecg,
        'slope': slope,
        'thal': thal
    }

    input_df = pd.DataFrame([input_dict])

    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols)

    for col in columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[columns]

    input_scaled = scaler.transform(input_encoded)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease (probability: {probability:.2%})")
    else:
        st.success(f"✅ Low risk of heart disease (probability: {probability:.2%})")