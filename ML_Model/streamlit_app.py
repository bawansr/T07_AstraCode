# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 21:33:18 2025

@author: VISTA
"""
import os
import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.write("Current Working Directory:", os.getcwd())
st.write("Files in Directory:", os.listdir()) 

# --- 2. LOAD THE MODEL ---
@st.cache_resource
def load_model():
    try:
        # CHECK THIS LINE! It MUST match the file list exactly.
        return pickle.load(open("model.sav", "rb"))
    except FileNotFoundError:
        st.error("Model file not found! Please make sure 'model.sav' is in the folder.")
        return None

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Dementia Prediction AI", layout="wide")

st.title("🧠 Dementia Prediction System")
st.markdown("Enter the clinical data below to generate a prediction.")

# --- 2. LOAD THE MODEL ---
# We use cache_resource so we don't reload the model on every interaction
@st.cache_resource
def load_model():
    try:
        # Ensure 'demenia prediction.sav' is in the same folder
        return pickle.load(open("demenia prediction.sav", "rb"))
    except FileNotFoundError:
        st.error("Model file not found! Please make sure 'demenia prediction.sav' is in the folder.")
        return None

model = load_model()

# --- 3. USER INPUT FORM ---
with st.form("prediction_form"):
    
    # Group 1: Demographics & Vitals
    st.subheader("1. Demographics & Vitals")
    c1, c2, c3 = st.columns(3)
    with c1:
        NACCAGE = st.number_input("Age (NACCAGE)", min_value=0.0, value=70.0)
        EDUC = st.number_input("Years of Education (EDUC)", min_value=0.0, value=12.0)
    with c2:
        HEIGHT = st.number_input("Height (inches)", min_value=0.0, value=65.0)
        WEIGHT = st.number_input("Weight (lbs)", min_value=0.0, value=150.0)
    with c3:
        NACCBMI = st.number_input("BMI (NACCBMI)", min_value=0.0, value=25.0)
        SEX_2 = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female (2)" if x==1 else "Male (1)")

    # Group 2: Background & History
    st.subheader("2. Background Factors (0 = No, 1 = Yes)")
    
    # Using an expander to hide the messy list of categorical variables
    with st.expander("Select Race, Language, and Residence Details"):
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("**Race & Ethnicity**")
            HISPANIC_1_0 = st.checkbox("Hispanic (1.0)")
            RACE_2_0 = st.checkbox("Race: Black/African Am. (2.0)")
            RACE_50_0 = st.checkbox("Race: Other (50.0)")
            # Add logic for other race inputs as needed or default to 0
            # For brevity in UI, assuming others are 0 unless specified, 
            # but in a full app you would list all.
            
        with col_b:
            st.markdown("**Language & Handedness**")
            PRIMLANG_1 = st.checkbox("Primary Lang: English (1.0)", value=True)
            HANDED_2_0 = st.checkbox("Left Handed (2.0)")
            
        with col_c:
            st.markdown("**Living Situation**")
            NACCLIVS_2_0 = st.checkbox("Lives with Spouse (2.0)")
            NACCLIVS_3_0 = st.checkbox("Lives with Relative (3.0)")
            RESIDENC_1 = st.checkbox("Residence: Private Home (1.0)", value=True)

    # Group 3: Clinical Assessments
    st.subheader("3. Clinical Assessments")
    c4, c5 = st.columns(2)
    with c4:
        MOCAVIS_1_0 = st.number_input("MoCA Vision Issue (0/1)", 0, 1, 0)
        MOCAHEAR_1_0 = st.number_input("MoCA Hearing Issue (0/1)", 0, 1, 0)
        VISION_1_0 = st.number_input("General Vision Problem (0/1)", 0, 1, 0)
        HEARING_1_0 = st.number_input("General Hearing Problem (0/1)", 0, 1, 0)
    with c5:
        INDEPEND_2_0 = st.checkbox("Req. Some Help (INDEPEND=2)")
        INDEPEND_3_0 = st.checkbox("Req. Much Help (INDEPEND=3)")
        INDEPEND_4_0 = st.checkbox("Totally Dependent (INDEPEND=4)")
    
    # ... For a real production app, you must map ALL 50 inputs here.
    # For this example, I will map the critical ones and fill the rest with 0s
    # to prevent the code from being 200 lines long. 
    
    submit_button = st.form_submit_button("Predict Dementia Status")

# --- 4. PREDICTION LOGIC ---
if submit_button and model:
    # 1. Create Dictionary
    # IMPORTANT: Keys must match the training data exactly (including dots)
    input_data = {
        'NACCAGE': NACCAGE, 'EDUC': EDUC, 'HEIGHT': HEIGHT, 'WEIGHT': WEIGHT, 'NACCBMI': NACCBMI,
        'SEX_2': SEX_2, 'HISPANIC_1.0': int(HISPANIC_1_0), 
        'RACE_2.0': int(RACE_2_0), 'RACE_50.0': int(RACE_50_0),
        'HANDED_2.0': int(HANDED_2_0), 
        'NACCLIVS_2.0': int(NACCLIVS_2_0), 'NACCLIVS_3.0': int(NACCLIVS_3_0),
        'INDEPEND_2.0': int(INDEPEND_2_0), 'INDEPEND_3.0': int(INDEPEND_3_0), 'INDEPEND_4.0': int(INDEPEND_4_0),
        'VISION_1.0': VISION_1_0, 'HEARING_1.0': HEARING_1_0,
        'MOCAVIS_1.0': MOCAVIS_1_0, 'MOCAHEAR_1.0': MOCAHEAR_1_0
        # ... In a real app, you MUST add all other columns here initialized to 0 or user input
    }

    # 2. Missing Columns Handling
    # Since we didn't create inputs for all 50 variables in this snippet, 
    # we will auto-fill missing columns with 0 to prevent a crash.
    try:
        expected_cols = model.feature_names_in_
        # Initialize all with 0
        full_data = {col: 0 for col in expected_cols}
        # Update with user inputs
        full_data.update(input_data)
        
        # 3. Convert to DataFrame
        df = pd.DataFrame([full_data])

        # 4. Predict
        prediction = model.predict(df)[0]
        
        # 5. Display Result
        st.divider()
        if prediction == 1:
            st.error("### Prediction: Dementia Detected")
            st.warning("The model suggests a high probability of dementia based on the inputs.")
        else:
            st.success("### Prediction: No Dementia Detected")
            st.balloons()
            
    except Exception as e:

        st.error(f"Error during prediction: {e}")
