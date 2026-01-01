#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Monatszahlen_Verkehrsunfälle.csv"
IMAGE_PATH = BASE_DIR / "image" / "visualization.png"

def preprocess_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")
    df = df[df['JAHR'] < 2021] # drop the records which come after 2020
    return df

# visualise historically the number of accidents per category (column1)
def plot_categories_over_time(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(20,10), facecolor='w')
    categories = df['MONATSZAHL'].unique()
    for cat in categories:
        sub_df = df[(df['MONATSZAHL'] == cat) & (df['AUSPRAEGUNG'] == 'insgesamt') & (pd.to_numeric(df['MONAT'], errors='coerce').notnull())].copy()
        sub_df.sort_values(by=['JAHR', 'MONAT'], inplace=True)
        ax.plot(pd.to_datetime(sub_df['MONAT'].astype(str), format='%Y%m'), sub_df['WERT'], label=cat)
    ax.legend()
    ax.set_title('Monthly Traffic Accidents Count per Category (until 2020)')
    ax.grid()
    plt.show()
    if not IMAGE_PATH.exists():
        plt.savefig(IMAGE_PATH)

if __name__ == "__main__":
    trafficAccidents = preprocess_data(DATA_PATH)
    plot_categories_over_time(trafficAccidents)
