#!/usr/bin/env python3

import pandas as pd
from prophet import Prophet
from visualization import preprocess_data, BASE_DIR, DATA_PATH
import pickle
import matplotlib.pyplot as plt

MODEL_PATH = BASE_DIR / "model"

# develop the prediction model for each category
def train_prophet_model(df: pd.DataFrame, category: str) -> Prophet:
    sub_df = df[(df['MONATSZAHL'] == category) & (df['AUSPRAEGUNG'] == 'insgesamt') & (pd.to_numeric(df['MONAT'], errors='coerce').notnull())].copy()
    sub_df.sort_values(by=['JAHR', 'MONAT'], inplace=True)
    sub_df['ds'] = pd.to_datetime(sub_df['MONAT'].astype(str), format='%Y%m') # Convert 'YYYY-MM' to the first day of that month
    sub_df['y'] = sub_df['WERT']
    model = Prophet(seasonality_mode='multiplicative', mcmc_samples=300)
    model.fit(sub_df[['ds', 'y']])
    return model

def save_model(model: Prophet, category: str):
    path = MODEL_PATH / f"prophet_model_{category}.pkl"
    if not path.exists():
        with open(path, 'wb') as f:
            pickle.dump(model, f)

def plot_prediction_vs_actual(model: Prophet, df: pd.DataFrame, category: str):
    sub_df = df[(df['MONATSZAHL'] == category) & (df['AUSPRAEGUNG'] == 'insgesamt') & (pd.to_numeric(df['MONAT'], errors='coerce').notnull())].copy()
    sub_df.sort_values(by=['JAHR', 'MONAT'], inplace=True)
    sub_df['ds'] = pd.to_datetime(sub_df['MONAT'].astype(str), format='%Y%m')
    sub_df['y'] = sub_df['WERT']
    future = model.make_future_dataframe(periods=12, freq='MS')
    forecast = model.predict(future)
    fig = model.plot(forecast)

def extract_2021_data(df: pd.DataFrame, category: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=",")
    df = df[df['JAHR'] == 2021] # keep only the records for 2021
    sub_df = df[(df['MONATSZAHL'] == category) & (df['AUSPRAEGUNG'] == 'insgesamt') & (pd.to_numeric(df['MONAT'], errors='coerce').notnull()) & (df['JAHR'] == 2021)].copy()
    sub_df.sort_values(by=['JAHR', 'MONAT'], inplace=True)
    sub_df['ds'] = pd.to_datetime(sub_df['MONAT'].astype(str), format='%Y%m')
    sub_df['y'] = sub_df['WERT']
    return sub_df

# compute the error between prediction values and the actual numbers (ground truth data)
def compute_prediction_error(model: Prophet, df: pd.DataFrame, category: str) -> float:
    sub_df = df[(df['MONATSZAHL'] == category) & (df['AUSPRAEGUNG'] == 'insgesamt') & (pd.to_numeric(df['MONAT'], errors='coerce').notnull())].copy()
    sub_df.sort_values(by=['JAHR', 'MONAT'], inplace=True)
    sub_df['ds'] = pd.to_datetime(sub_df['MONAT'].astype(str), format='%Y%m')
    sub_df['y'] = sub_df['WERT']
    future = model.make_future_dataframe(periods=12, freq='MS')
    forecast = model.predict(future)
    merged = pd.merge(sub_df, forecast[['ds', 'yhat']], on='ds', how='left')
    merged.dropna(subset=['yhat'], inplace=True)
    mse = ((merged['y'] - merged['yhat']) ** 2).mean()
    return mse

if __name__ == "__main__":
    trafficAccidents = preprocess_data(DATA_PATH)
    categories = trafficAccidents['MONATSZAHL'].unique()
    for cat in categories:
        model = train_prophet_model(trafficAccidents, cat)
        save_model(model, cat)
        plot_prediction_vs_actual(model, trafficAccidents, cat)
        error = compute_prediction_error(model, extract_2021_data(trafficAccidents, cat), cat)
        print(f"Category: {cat}, MSE for 2021 predictions: {error}")
    plt.show()