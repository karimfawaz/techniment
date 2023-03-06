# Import necessary libraries
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Get the OHLCV data from the API
url = 'https://min-api.cryptocompare.com/data/histominute'
params = {'fsym': 'BTC', 'tsym': 'USD', 'limit': 2000}
response = requests.get(url, params=params)
data = response.json()

# Create a dataframe from the OHLCV data
df = pd.DataFrame(data['Data'])
df.set_index('time', inplace=True)

# Calculate the technical indicators
df['7-day SMA'] = df['close'].rolling(window=7).mean()
df['21-day SMA'] = df['close'].rolling(window=21).mean()
df['EMA_0.67'] = df['close'].ewm(alpha=0.67).mean()
df['12-day EMA'] = df['close'].ewm(span=12).mean()
df['26-day EMA'] = df['close'].ewm(span=26).mean()
df['MACD'] = df['12-day EMA'] - df['26-day EMA']
df['20-day STD'] = df['close'].rolling(window=20).std()
df['Upper BB'] = df['21-day SMA'] + (2 * df['20-day STD'])
df['Lower BB'] = df['21-day SMA'] - (2 * df['20-day STD'])
df['High-Low Spread'] = df['high'] - df['low']

# Get the Ethereum and Gold spot prices
url_eth = 'https://min-api.cryptocompare.com/data/price'
params_eth = {'fsym': 'ETH', 'tsyms': 'USD'}
response_eth = requests.get(url_eth, params=params_eth)
eth_price = response_eth.json()['USD']

url_gold = 'https://min-api.cryptocompare.com/data/price'
params_gold = {'fsym': 'XAU', 'tsyms': 'USD'}
response_gold = requests.get(url_eth, params=params_gold)
gold_price = response_gold.json()['USD']


# Add the Ethereum and Gold spot prices to the dataframe
df['ETH Price'] = eth_price
df['Gold Price'] = gold_price

# Calculate the moving average indicator
df['MA Indicator'] = np.where(df['7-day SMA'] > df['21-day SMA'], 1, 0)

# Create the feature and target arrays
X = df.drop(['close'], axis=1).values
y = df['close'].values

# Split the data into training, evaluation, and testing sets
X_train, X_eval, y_train, y_eval = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_eval, X_test, y_eval, y_test = train_test_split(
    X_eval, y_eval, test_size=0.5, random_state=42)

# Create the Support Vector Model
model = SVR(kernel='rbf', gamma='scale')

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the evaluation and testing sets
y_pred_eval = model.predict(X_eval)
y_pred_test = model.predict(X_test)

# Calculate the evaluation and testing scores
eval_score = mean_squared_error(y_eval, y_pred_eval)
test_score = mean_squared_error(y_test, y_pred_test)

# Print the evaluation and testing scores
print('Evaluation Score:', eval_score)
print('Testing Score:', test_score)
