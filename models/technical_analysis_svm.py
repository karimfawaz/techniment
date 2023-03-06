from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import requests

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
print(df)

# Calculate simple moving averages
df['sma5'] = df['close'].rolling(5).mean()
df['sma10'] = df['close'].rolling(10).mean()
df['sma20'] = df['close'].rolling(20).mean()

# Calculate exponential moving averages
df['ema5'] = df['close'].ewm(span=5).mean()
df['ema10'] = df['close'].ewm(span=10).mean()
df['ema20'] = df['close'].ewm(span=20).mean()

# Calculate the MACD indicator
df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
df['macd_signal'] = df['macd'].ewm(span=9).mean()
df['macd_hist'] = df['macd'] - df['macd_signal']


# Calculate the moving average indicator
df['ma'] = df['close'].rolling(10).mean()

# Calculate Bollinger bands
df['sma'] = df['close'].rolling(20).mean()
df['std'] = df['close'].rolling(20).std()
df['upper_band'] = df['sma'] + 2 * df['std']
df['lower_band'] = df['sma'] - 2 * df['std']

# Calculate the high-low spread
df['spread'] = df['high'] - df['low']

df = df.dropna()


# Create the target column
df['target'] = df['close'].diff().shift(-1)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Split the data into training and testing sets
X = df[['sma5', 'sma10', 'sma20', 'ema5', 'ema10', 'ema20', 'macd',
        'macd_signal', 'ma', 'upper_band', 'lower_band', 'spread']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Build the SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate the model on the testing set
accuracy = model.score(X_test, y_test)
print(f'Test accuracy: {accuracy:.2f}')

# Make predictions on the testing set
y_pred = model.predict(X_test)
