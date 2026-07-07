import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("🚢 Titanic Survival Predictor AI")
st.write("Afzal Bhai ki Pehli AI Web App!")

pclass = st.selectbox("Passenger Class:", [1, 2, 3], index=2)
age = st.slider("Age (Umar):", 1, 100, 46)
fare = st.slider("Ticket Fare ($):", 0.0, 500.0, 30.0)
gender = st.selectbox("Gender:", ["Male", "Female"])
sex_male = 1 if gender == "Male" else 0

X_dummy = pd.DataFrame([
    [3, 22.0, 7.25, 1], [1, 38.0, 71.28, 0], [3, 26.0, 7.92, 0],
    [1, 35.0, 53.10, 0], [3, 35.0, 8.05, 1]
], columns=["Pclass", "Age", "Fare", "Sex_male"])
y_dummy = [0, 1, 1, 1, 0]

model = LogisticRegression()
model.fit(X_dummy, y_dummy)

if st.button("🔮 Predict (Andaza Lagayein)"):
    # Double brackets [ [ ... ] ] use karein
    input_data = pd.DataFrame([[pclass, age, fare, sex_male]], 
                              columns=["Pclass", "Age", "Fare", "Sex_male"])
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    
    if prediction == 1:
        st.success(f"✅ Ye shakhs zinda bach gaya! (Probability: {prob[1]*100:.2f}%)")
    else:
        st.error(f"❌ Ye shakhs doob gaya. (Probability: {prob[1]*100:.2f}%)")
