#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pickle
from pathlib import Path
import sys

# add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visualization import preprocess_data, DATA_PATH
from predictor import predict_accidents
from model_trainer import train_prophet_model, save_model, MODEL_PATH

app = FastAPI(title="Traffic Accident Predictor", version="1.0.0")


class PredictionRequest(BaseModel):
    """Request body for accident prediction"""
    year: int
    month: int
    category: str = "Alkoholunfälle"
    type: str = "insgesamt"


class PredictionResponse(BaseModel):
    """Response body with prediction result"""
    prediction: float


@app.get("/")
def read_root():
    """Health check endpoint"""
    return {"message": "Traffic Accident Predictor API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    """
    Predict traffic accidents for a given year and month.
    
    Request:
        - year: int (e.g., 2020)
        - month: int (1-12)
        - category: str (default: "Alkoholunfälle")
        - type: str (default: "insgesamt")
    
    Returns:
        - prediction: float (predicted accident count)
    """
    try:
        # Validate month
        if not 1 <= request.month <= 12:
            raise ValueError("Month must be between 1 and 12")
        
        if request.year < 2000 or request.year > 2100:
            raise ValueError("Year must be between 2000 and 2100")
        
        if not request.type == "insgesamt":
            raise ValueError("Currently, only 'insgesamt' type is supported")
        
        if request.category not in ["Alkoholunfälle", "Fluchtunfälle", "Verkehrsunfälle"]:
            raise ValueError(f"Category '{request.category}' is not recognized")

        # Load model
        model_path = MODEL_PATH / f"prophet_model_{request.category}.pkl"
        if not model_path.exists():
            trafficAccidents = preprocess_data(DATA_PATH)
            model = train_prophet_model(trafficAccidents, request.category)
            save_model(model, request.category)
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Make prediction
        predicted_value = predict_accidents(
            model=model,
            category=request.category,
            year=request.year,
            month=request.month,
            type=request.type
        )
        
        return PredictionResponse(
            prediction=predicted_value,
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)