import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model/model_student.pkl")

st.title("🎓 Student Status Predictor")

avg_grade = st.number_input("Average Grade", 0.0, 20.0, 12.0)
total_approval_rate = st.slider("Total Approval Rate", 0.0, 1.0, 0.75)
is_first_choice = st.selectbox("First Choice?", ["Yes", "No"])
parents_education_score = st.slider("Parents' Education Score", 0, 100, 50)
financial_stress = st.selectbox("Financial Stress?", ["Yes", "No"])
student_segment = st.number_input("Student Segment (numeric)", 0, 10, 3)

is_first_choice_encoded = 1 if is_first_choice == "Yes" else 0
financial_stress_encoded = 1 if financial_stress == "Yes" else 0

input_data = pd.DataFrame([{
    "avg_grade": avg_grade,
    "total_approval_rate": total_approval_rate,
    "is_first_choice": is_first_choice_encoded,
    "parents_education_score": parents_education_score,
    "financial_stress": financial_stress_encoded,
    "student_segment": student_segment
}])

if st.button("Predict Student Status"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][model.classes_.tolist().index(prediction)]

    if prediction.lower() == "graduate":
        st.success(f"✅ Prediction: Graduate (Confidence: {prob:.2f})")
    elif prediction.lower() == "dropout":
        st.error(f"🚨 Prediction: Dropout (Confidence: {prob:.2f})")
    else:
        st.info(f"Prediction: {prediction} (Confidence: {prob:.2f})")
