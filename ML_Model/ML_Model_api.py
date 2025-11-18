# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import List # Needed for batch inputs
import pickle
import pandas as pd
import numpy as np

# Global variable to hold the model
models = {}

# --- 1. LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("Loading model...")
        with open("demenia prediction.sav", "rb") as f:
            models["classifier"] = pickle.load(f)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: 'demenia prediction.sav' not found.")
        models["classifier"] = None
    yield
    models.clear()

app = FastAPI(
    title="Dementia Prediction API",
    description="API for predicting Dementia status based on clinical data.",
    version="1.0.0",
    lifespan=lifespan
)

# --- 2. DATA MODEL ---
class ModelInput(BaseModel):
    # Continuous Variables
    NACCAGE: float
    EDUC: float
    HEIGHT: float
    WEIGHT: float
    NACCBMI: float

    # Categorical Variables
    SEX_2: int
    HISPANIC_1_0: int = Field(alias="HISPANIC_1.0")
    RACE_2_0: int = Field(alias="RACE_2.0")
    RACE_3_0: int = Field(alias="RACE_3.0")
    RACE_4_0: int = Field(alias="RACE_4.0")
    RACE_5_0: int = Field(alias="RACE_5.0")
    RACE_50_0: int = Field(alias="RACE_50.0")
    HANDED_2_0: int = Field(alias="HANDED_2.0")
    HANDED_3_0: int = Field(alias="HANDED_3.0")
    PRIMLANG_2_0: int = Field(alias="PRIMLANG_2.0")
    PRIMLANG_3_0: int = Field(alias="PRIMLANG_3.0")
    PRIMLANG_4_0: int = Field(alias="PRIMLANG_4.0")
    PRIMLANG_5_0: int = Field(alias="PRIMLANG_5.0")
    PRIMLANG_6_0: int = Field(alias="PRIMLANG_6.0")
    PRIMLANG_8_0: int = Field(alias="PRIMLANG_8.0")
    MARISTAT_2_0: int = Field(alias="MARISTAT_2.0")
    MARISTAT_3_0: int = Field(alias="MARISTAT_3.0")
    MARISTAT_4_0: int = Field(alias="MARISTAT_4.0")
    MARISTAT_5_0: int = Field(alias="MARISTAT_5.0")
    MARISTAT_6_0: int = Field(alias="MARISTAT_6.0")
    RESIDENC_2_0: int = Field(alias="RESIDENC_2.0")
    RESIDENC_3_0: int = Field(alias="RESIDENC_3.0")
    RESIDENC_4_0: int = Field(alias="RESIDENC_4.0")
    NACCLIVS_2_0: int = Field(alias="NACCLIVS_2.0")
    NACCLIVS_3_0: int = Field(alias="NACCLIVS_3.0")
    NACCLIVS_4_0: int = Field(alias="NACCLIVS_4.0")
    NACCLIVS_5_0: int = Field(alias="NACCLIVS_5.0")
    INDEPEND_2_0: int = Field(alias="INDEPEND_2.0")
    INDEPEND_3_0: int = Field(alias="INDEPEND_3.0")
    INDEPEND_4_0: int = Field(alias="INDEPEND_4.0")
    NACCFAM_1_0: int = Field(alias="NACCFAM_1.0")
    NACCMOM_1_0: int = Field(alias="NACCMOM_1.0")
    NACCDAD_1_0: int = Field(alias="NACCDAD_1.0")
    NACCFFTD_1: int
    VISION_1_0: int = Field(alias="VISION_1.0")
    HEARING_1_0: int = Field(alias="HEARING_1.0")
    INRELTO_2_0: int = Field(alias="INRELTO_2.0")
    INRELTO_3_0: int = Field(alias="INRELTO_3.0")
    INRELTO_4_0: int = Field(alias="INRELTO_4.0")
    INRELTO_5_0: int = Field(alias="INRELTO_5.0")
    INRELTO_6_0: int = Field(alias="INRELTO_6.0")
    INRELTO_7_0: int = Field(alias="INRELTO_7.0")
    INRELY_1_0: int = Field(alias="INRELY_1.0")
    NACCREAS_2_0: int = Field(alias="NACCREAS_2.0")
    NACCREAS_7_0: int = Field(alias="NACCREAS_7.0")
    NACCREFR_2_0: int = Field(alias="NACCREFR_2.0")
    NACCREFR_8_0: int = Field(alias="NACCREFR_8.0")
    MOCAVIS_1_0: int = Field(alias="MOCAVIS_1.0")
    MOCAHEAR_1_0: int = Field(alias="MOCAHEAR_1.0")

    class Config:
        populate_by_name = True

# --- 3. API ENDPOINTS ---

# Endpoint 1: Health Check
@app.get("/health")
def health_check():
    """Checks if the server is running and the model is loaded."""
    if models["classifier"] is not None:
        return {"status": "Online", "model_status": "Loaded"}
    else:
        return {"status": "Online", "model_status": "Failed to Load"}

# Endpoint 2: Model Metadata
@app.get("/model_info")
def model_info():
    """Returns the feature names the model expects."""
    if models["classifier"] is None:
         raise HTTPException(status_code=500, detail="Model not loaded")
    
    # If your model is XGBoost/Sklearn, it likely has feature_names_in_
    try:
        features = list(models["classifier"].feature_names_in_)
        return {
            "model_type": type(models["classifier"]).__name__,
            "expected_features": features,
            "total_features": len(features)
        }
    except AttributeError:
        return {"message": "Model does not support feature name extraction."}


# Endpoint 3: Single Prediction (The one you had)
@app.post('/predict_dementia')
def predict_dementia(input_data: ModelInput):
    if models["classifier"] is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        input_dict = input_data.model_dump(by_alias=True)
        df = pd.DataFrame([input_dict])
        prediction = models["classifier"].predict(df)
        
        result_class = int(prediction[0]) 
        label = "Dementia" if result_class == 1 else "No Dementia"
        
        return {
            "prediction_class": result_class,
            "prediction_label": label
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Endpoint 4: BATCH Prediction (Process multiple people at once)
@app.post('/predict_batch')
def predict_batch(inputs: List[ModelInput]):
    """Predicts dementia for a list of people."""
    if models["classifier"] is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # 1. Convert list of Pydantic objects to list of dicts
        data_list = [item.model_dump(by_alias=True) for item in inputs]
        
        # 2. Create ONE DataFrame for all rows
        df = pd.DataFrame(data_list)
        
        # 3. Predict all at once (Very fast)
        predictions = models["classifier"].predict(df)
        
        # 4. Format results
        results = []
        for i, pred in enumerate(predictions):
            result_class = int(pred)
            label = "Dementia" if result_class == 1 else "No Dementia"
            results.append({
                "row_index": i,
                "prediction_label": label,
                "prediction_class": result_class
            })
            
        return {"batch_results": results}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))