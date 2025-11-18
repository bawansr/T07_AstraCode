import streamlit as st
import pandas as pd
import pickle
from pathlib import Path # <--- CRITICAL FIX: We need this to use Path

# 1. APP CONFIG
st.set_page_config(page_title="Dementia AI", layout="wide")
st.title("🧠 Dementia Prediction System")

# 2. LOAD MODEL (Cached so it doesn't reload every click)
@st.cache_resource
def load_model():
    # Because your Streamlit file is located INSIDE the ML_Model folder
    # the model file should be accessed directly by name.
    MODEL_FILENAME = "demenia prediction.sav"
    
    # We use pathlib to build the path robustly
    # We use .resolve() just to ensure we get the absolute path used for opening
    MODEL_PATH = Path(MODEL_FILENAME) 
    
    st.info(f"Attempting to load model from: {MODEL_PATH.resolve()}")

    try:
        # Open the file using the resolved path
        with open(MODEL_PATH.resolve(), "rb") as f:
            return pickle.load(f)
            
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Model file not found. Please check that '{MODEL_FILENAME}' is directly next to 'streamlit_app.py'.")
        return None

# THIS CALL IS MADE IMMEDIATELY ON STARTUP
model = load_model() 

# 3. USER INTERFACE (Only shows if 'model' loaded successfully)
if model: 
    with st.form("prediction_form"):
        st.subheader("Patient Vitals")
        
        # --- INPUTS ---
        c1, c2 = st.columns(2)
        with c1:
            NACCAGE = st.number_input("Age (Years)", value=70.0)
            EDUC = st.number_input("Education Years", value=12.0)
        with c2:
            NACCBMI = st.number_input("BMI", value=25.0)
            # Example of handling a categorical feature from the data
            MOCAHEAR_1_0 = st.checkbox("Has MoCA Hearing Issues", value=False)

        # --- SUBMIT BUTTON ---
        submit = st.form_submit_button("Predict Dementia Risk", type="primary")

    # 4. PREDICTION LOGIC
    if submit:
        # Create dictionary matching training data keys exactly
        input_data = {
            'NACCAGE': NACCAGE,
            'EDUC': EDUC,
            'NACCBMI': NACCBMI,
            'MOCAHEAR_1.0': int(MOCAHEAR_1_0),
            # Add all other required features here, e.g., 'SEX_2': 1, 'RACE_1.0': 0, etc.
            # You must include all 50+ features your model was trained on.
        }
        
        # Safety feature: Ensure all expected columns are present, filling others with 0
        expected_cols = model.feature_names_in_
        full_data = {col: 0 for col in expected_cols}
        full_data.update(input_data)
        
        # Predict
        df = pd.DataFrame([full_data])
        pred = model.predict(df)[0]
        
        st.divider()
        if pred == 1:
            st.error("⚠️ Prediction: HIGH RISK (Likely Dementia Detected)")
        else:
            st.success("✅ Prediction: LOW RISK (No Dementia Detected)")
