import streamlit as st
import pandas as pd
import pickle

st.title("🚢 Titanic Survival Predictor AI")
st.write("Afzal Bhai ki Asli AI Web App!")

# User inputs
pclass = st.selectbox("Passenger Class:", [1, 2, 3], index=2)
age = st.slider("Age (Umar):", 1, 100, 46)
fare = st.slider("Ticket Fare ($):", 0.0, 500.0, 30.0)
gender = st.selectbox("Gender:", ["Male", "Female"])
sex_male = 1 if gender == "Male" else 0

# 🔥 ASLI MODEL KO FILE SE LOAD KARNA
with open('titanic_model.pkl', 'rb') as file:
    asli_model = pickle.load(file)

if st.button("🔮 Predict (Andaza Lagayein)"):
    input_data = pd.DataFrame([[pclass, age, fare, sex_male]], 
                              columns=["Pclass", "Age", "Fare", "Sex_male"])
    
    # Asli model se prediction aur probability nikalna
    prediction = asli_model.predict(input_data)[0]
    prob = asli_model.predict_proba(input_data)[0]
    
    if prediction == 1:
        st.success(f"✅ Ye shakhs zinda bach gaya! (Probability: {prob[1]*100:.2f}%)")
    else:
        st.error(f"❌ Ye shakhs doob gaya. (Probability: {prob[1]*100:.2f}%)")
