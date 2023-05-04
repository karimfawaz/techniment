import pandas as pd
import numpy as np
import requests
import pickle

# Get the OHLCV data from the API
API_URL = "https://min-api.cryptocompare.com/data/histohour"
symbol = "BTC"
limit = 2000

params = {
    "fsym": symbol,
    "tsym": "USD",
    "limit": limit,
}

response = requests.get(API_URL, params=params)
data = response.json()["Data"]

df = pd.DataFrame(
    data, columns=["time", "open", "high", "low", "close", "volumefrom", "volumeto"]
)
df["time"] = pd.to_datetime(df["time"], unit="s")

# Convert the 'time' column to a Unix timestamp
df["timestamp"] = pd.to_datetime(df["time"]).astype(np.int64) // 10**9
df = df.drop(columns=["time"])  # Remove the 'time' column

# Map the column names in the new data to match the original data
df = df.rename(columns={"volumefrom": "Volume BTC", "volumeto": "Volume USD"})

# Add the 'unix' column as a copy of the 'timestamp' column
df["unix"] = df["timestamp"]

print(df)
