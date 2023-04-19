import pandas as pd
import numpy as np
import requests
import pickle
from sklearn.metrics import accuracy_score, classification_report

# Get the OHLCV data from the API
API_URL = 'https://min-api.cryptocompare.com/data/histohour'
symbol = 'BTC'
limit = 2000

params = {
    'fsym': symbol,
    'tsym': 'USD',
    'limit': limit,
}

response = requests.get(API_URL, params=params)
data = response.json()['Data']
df = pd.DataFrame(data, columns=[
                  'time', 'open', 'high', 'low', 'close', 'volumefrom', 'volumeto'])
df['time'] = pd.to_datetime(df['time'], unit='s')

# Convert the 'time' column to a Unix timestamp
df['timestamp'] = pd.to_datetime(df['time']).astype(np.int64) // 10**9
df = df.drop(columns=['time'])  # Remove the 'time' column

# Map the column names in the new data to match the original data
df = df.rename(columns={'volumefrom': 'Volume BTC', 'volumeto': 'Volume USD'})

# Add the 'unix' column as a copy of the 'timestamp' column
df['unix'] = df['timestamp']

# Add the technical indicators
df['7-day SMA'] = df['close'].rolling(7).mean()
df['21-day SMA'] = df['close'].rolling(21).mean()
df['EMA_0.67'] = df['close'].ewm(alpha=0.67).mean()
df['12-day EMA'] = df['close'].ewm(span=12).mean()
df['26-day EMA'] = df['close'].ewm(span=26).mean()
df['MACD'] = df['12-day EMA'] - df['26-day EMA']
df['20-day STD'] = df['close'].rolling(window=20).std()
df['Upper BB'] = df['21-day SMA'] + (2 * df['20-day STD'])
df['Lower BB'] = df['21-day SMA'] - (2 * df['20-day STD'])
df['High-Low Spread'] = df['high'] - df['low']
df['MA Indicator'] = np.where(df['7-day SMA'] > df['21-day SMA'], 1, 0)

# Calculate the price difference between consecutive periods
df['Price Diff'] = df['close'].diff()

# Create a binary variable that indicates whether the price has gone up (1) or down (0)
df['Price Direction'] = np.where(df['Price Diff'] > 0, 1, 0)

# Drop rows with NaN values
df = df.dropna()

# Load the pickled model and scaler
with open("model/best_svm.pkl", "rb") as f:
    loaded_model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    loaded_scaler = pickle.load(f)

# Scale the new data
new_X = df.drop(columns=['Price Direction'])

# Reorder the columns in the new data to match the original data
column_order = ['unix', 'open', 'high', 'low', 'close', 'Volume BTC', 'Volume USD', 'timestamp', '7-day SMA', '21-day SMA', 'EMA_0.67',
                '12-day EMA', '26-day EMA', 'MACD', '20-day STD', 'Upper BB', 'Lower BB', 'High-Low Spread', 'MA Indicator', 'Price Diff']
new_X = new_X[column_order]

# Scale the new data
new_X_scaled = loaded_scaler.transform(new_X)

# Make predictions on the new data
y_new_pred = loaded_model.predict(new_X_scaled)
y_new = df['Price Direction']
# Add the predictions to the dataframe
df['Predicted Price Direction'] = y_new_pred


# Compute the accuracy percentage
accuracy = accuracy_score(y_new, y_new_pred)
print(f"Accuracy on new data: {accuracy:.4f}")
