# Import necessary libraries
import requests
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Get the OHLCV data from the API

API_URL = 'https://min-api.cryptocompare.com/data/histominute'

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

df.set_index('time', inplace=True)

# Calculate the technical indicators
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

df = df.fillna(0)


# Get Ethereum and Gold spot prices
url_eth = 'https://min-api.cryptocompare.com/data/price?fsym=ETH&tsyms=USD'
data_eth = requests.get(url_eth).json()
url_gold = 'https://min-api.cryptocompare.com/data/price?fsym=XAU&tsyms=USD'
data_gold = requests.get(url_gold).json()

# Add Ethereum and Gold spot prices to dataframe
df['Ethereum Price'] = data_eth['USD']
df['Gold Spot Price'] = data_gold['USD']

# Calculate the moving average indicator
df['MA Indicator'] = np.where(df['7-day SMA'] > df['21-day SMA'], 1, 0)

# Split the data into training, evaluation, and testing sets
train_data = df.iloc[:int(len(df)*0.6)]
eval_data = df.iloc[int(len(df)*0.6):int(len(df)*0.8)]
test_data = df.iloc[int(len(df)*0.8):]

# Create the feature and target arrays
X_train = train_data.drop('close', axis=1).values
y_train = train_data['close'].values
X_eval = eval_data.drop('close', axis=1).values
y_eval = eval_data['close'].values
X_test = test_data.drop('close', axis=1).values
y_test = test_data['close'].values


# Create and fit the Support Vector Model
svr = SVR()
svr.fit(X_train, y_train)


# Make predictions on the evaluation and testing sets
y_eval_pred = svr.predict(X_eval)
y_test_pred = svr.predict(X_test)

# Calculate the evaluation and testing scores
eval_score = mean_squared_error(y_eval, y_eval_pred)
test_score = mean_squared_error(y_test, y_test_pred)

print('Evaluation Score:', eval_score)
print('Testing Score:', test_score)
