#!/usr/bin/env python3

import pandas as pd
import pickle
from prophet import Prophet
from model_trainer import MODEL_PATH

def predict_accidents(model, category='Alkoholunfälle', year=2021, month=1, type='insgesamt') -> float:
    # create a dataframe for the prediction date
    pred_date = pd.to_datetime(f"{year}-{month:02d}-01")
    future = pd.DataFrame({'ds': [pred_date]})

    # make prediction
    forecast = model.predict(future)
    predicted_value = forecast['yhat'].values[0]

    print(f"Predicted value for category '{category}', type '{type}', year {year}, month {month:02d}: {predicted_value}")
    
    return predicted_value

# forecasts the values for:
# {
# Category: 'Alkoholunfälle'
# Type: 'insgesamt'
# Year: '2021'
# Month: '01'
# }
if __name__ == "__main__":

    category = 'Alkoholunfälle'
    year = 2021
    month = 1
    type = 'insgesamt'

    # check model exists
    model_path = MODEL_PATH / f"prophet_model_{category}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model for category {category} not found at {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    predict_accidents(model, category, year, month, type)
    