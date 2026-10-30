import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('heart_LG.pkl')
columns = joblib.load('columns.pkl')

st.title('Heart Disease Prediction By Huzaifa ❤️')
st.markdown('Provide the following information to predict the chances of heart disease')

# -----------------------------
# User Inputs
# -----------------------------

age = st.slider('Age', 18, 100, 40)

sex = st.selectbox(
    'Sex',
    ['Male', 'Female']
)

cp = st.selectbox(
    'Chest Pain Type',
    [
        'Typical Angina',
        'Atypical Angina',
        'Non-Anginal Pain',
        'Asymptomatic'
    ]
)

trestbps = st.slider(
    'Resting Blood Pressure (in mm Hg)',
    80, 200, 120
)

chol = st.slider(
    'Serum Cholesterol (in mg/dl)',
    100, 600, 200
)

fbs = st.selectbox(
    'Fasting Blood Sugar > 120 mg/dl',
    ['No', 'Yes']
)

restecg = st.selectbox(
    'Resting Electrocardiographic Results',
    [
        'Normal',
        'ST-T Wave Abnormality',
        'Left Ventricular Hypertrophy'
    ]
)

thalach = st.slider(
    'Maximum Heart Rate Achieved',
    60, 220, 150
)

exang = st.selectbox(
    'Exercise Induced Angina',
    ['No', 'Yes']
)

oldpeak = st.slider(
    'ST Depression Induced by Exercise Relative to Rest',
    0.0, 10.0, 1.0
)

slope = st.selectbox(
    'Slope of the Peak Exercise ST Segment',
    [
        'Upsloping',
        'Flat',
        'Downsloping'
    ]
)

ca = st.selectbox(
    'Number of Major Vessels (0-4)',
    [0, 1, 2, 3, 4]
)

thal = st.selectbox(
    'Thalassemia',
    [
        'Normal',
        'Fixed Defect',
        'Reversible Defect'
    ]
)


# -----------------------------
# Prediction
# -----------------------------

if st.button('Predict'):

    # Convert categorical values to the same
    # numeric values used in the dataset

    sex_value = 1 if sex == 'Male' else 0

    cp_mapping = {
        'Typical Angina': 0,
        'Atypical Angina': 1,
        'Non-Anginal Pain': 2,
        'Asymptomatic': 3
    }

    fbs_value = 1 if fbs == 'Yes' else 0

    restecg_mapping = {
        'Normal': 0,
        'ST-T Wave Abnormality': 1,
        'Left Ventricular Hypertrophy': 2
    }

    exang_value = 1 if exang == 'Yes' else 0

    slope_mapping = {
        'Upsloping': 0,
        'Flat': 1,
        'Downsloping': 2
    }

    thal_mapping = {
        'Normal': 1,
        'Fixed Defect': 2,
        'Reversible Defect': 3
    }

    # Create numeric input
    raw_input = {
        'age': age,
        'sex': sex_value,
        'cp': cp_mapping[cp],
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs_value,
        'restecg': restecg_mapping[restecg],
        'thalach': thalach,
        'exang': exang_value,
        'oldpeak': oldpeak,
        'slope': slope_mapping[slope],
        'ca': ca,
        'thal': thal_mapping[thal]
    }

    # DataFrame
    input_df = pd.DataFrame([raw_input])

    # Make sure columns are exactly in training order
    input_df = input_df[columns]

    # Prediction
    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    # Result
    if prediction == 1:

        st.error(
            f'The model predicts a high chance of heart disease.'
        )

        st.write(
            f'Estimated probability: {probability * 100:.2f}%'
        )

    else:

        st.success(
            f'The model predicts a low chance of heart disease.'
        )

        st.write(
            f'Estimated probability: {probability * 100:.2f}%'
        )