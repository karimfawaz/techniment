import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import requests


def create_ta_svm():
    url = "./data/Bitstamp_BTCUSD_1h.csv"
    data = pd.read_csv(url)

    # Convert the date column to a Unix timestamp
    data["timestamp"] = pd.to_datetime(data["date"]).astype(np.int64) // 10**9
    # Remove the date and symbol columns
    data = data.drop(columns=["date", "symbol"])

    data["7-day SMA"] = data["close"].rolling(7).mean()
    data["21-day SMA"] = data["close"].rolling(21).mean()
    data["EMA_0.67"] = data["close"].ewm(alpha=0.67).mean()
    data["12-day EMA"] = data["close"].ewm(span=12).mean()
    data["26-day EMA"] = data["close"].ewm(span=26).mean()
    data["MACD"] = data["12-day EMA"] - data["26-day EMA"]
    data["20-day STD"] = data["close"].rolling(window=20).std()
    data["Upper BB"] = data["21-day SMA"] + (2 * data["20-day STD"])
    data["Lower BB"] = data["21-day SMA"] - (2 * data["20-day STD"])
    data["High-Low Spread"] = data["high"] - data["low"]
    data["MA Indicator"] = np.where(data["7-day SMA"] > data["21-day SMA"], 1, 0)

    # Calculate the price difference between consecutive periods
    data["Price Diff"] = data["close"].diff()

    # Create a binary variable that indicates whether the price has gone up (1) or down (0)
    data["Price Direction"] = np.where(data["Price Diff"] > 0, 1, 0)

    # Drop rows with NaN values
    data = data.dropna()

    # Sort date from oldest to newest unix timestamp
    data = data.sort_values("timestamp", ascending=True)
    data.to_csv("./data/ta_processed_data.csv", index=False)

    # Set the features (X) and target (y) variables
    X = data.drop(columns=["Price Direction"])
    y = data["Price Direction"]

    # Calculate the indices for splitting the data
    train_size = int(len(data) * 0.55)
    val_size = int(len(data) * 0.2)

    # Split the data into training, validation, and testing sets
    X_train, X_val, X_test = (
        X[:train_size],
        X[train_size : train_size + val_size],
        X[train_size + val_size :],
    )

    # print(X_train.columns)

    y_train, y_val, y_test = (
        y[:train_size],
        y[train_size : train_size + val_size],
        y[train_size + val_size :],
    )

    # Apply feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Define the parameter grid for GridSearchCV
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
    }

    # Use GridSearchCV to find the best hyperparameters
    grid = GridSearchCV(LinearSVC(), param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train_scaled, y_train)

    # print("Best parameters found: ", grid.best_params_)

    # Train the model with the best hyperparameters
    best_svm = LinearSVC(C=grid.best_params_["C"])
    best_svm.fit(X_train_scaled, y_train)

    # Make predictions on the validation set
    y_val_pred = best_svm.predict(X_val_scaled)

    # Calculate the accuracy and print the classification report for the validation set
    val_accuracy = accuracy_score(y_val, y_val_pred)
    # print(f"Validation Accuracy: {val_accuracy:.4f}")
    # print(classification_report(y_val, y_val_pred))

    # Make predictions on the test set
    y_test_pred = best_svm.predict(X_test_scaled)

    # Calculate the accuracy and print the classification report for the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    # print(f"Test Accuracy: {test_accuracy:.4f}")
    # print(classification_report(y_test, y_test_pred))

    # Save the trained model and scaler to disk
    with open("model/best_svm.pkl", "wb") as file:
        pickle.dump(best_svm, file)
    with open("model/scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)


def get_new_data():
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
    print(data)
    df = pd.DataFrame(
        data, columns=["time", "open", "high", "low", "close", "volumefrom", "volumeto"]
    )
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # Convert the 'time' column to a Unix timestamp
    df["timestamp"] = pd.to_datetime(df["time"]).astype(np.int64) // 10**9

    # Map the column names in the new data to match the original data
    df = df.rename(columns={"volumefrom": "Volume BTC", "volumeto": "Volume USD"})

    # Add the 'unix' column as a copy of the 'timestamp' column
    df["unix"] = df["timestamp"]

    # Add the technical indicators
    df["7-day SMA"] = df["close"].rolling(7).mean()
    df["21-day SMA"] = df["close"].rolling(21).mean()
    df["EMA_0.67"] = df["close"].ewm(alpha=0.67).mean()
    df["12-day EMA"] = df["close"].ewm(span=12).mean()
    df["26-day EMA"] = df["close"].ewm(span=26).mean()
    df["MACD"] = df["12-day EMA"] - df["26-day EMA"]
    df["20-day STD"] = df["close"].rolling(window=20).std()
    df["Upper BB"] = df["21-day SMA"] + (2 * df["20-day STD"])
    df["Lower BB"] = df["21-day SMA"] - (2 * df["20-day STD"])
    df["High-Low Spread"] = df["high"] - df["low"]
    df["MA Indicator"] = np.where(df["7-day SMA"] > df["21-day SMA"], 1, 0)

    # Calculate the price difference between consecutive periods
    df["Price Diff"] = df["close"].diff()

    # Create a binary variable that indicates whether the price has gone up (1) or down (0)
    df["Price Direction"] = np.where(df["Price Diff"] > 0, 1, 0)

    # Drop rows with NaN values
    df = df.dropna()

    return df


def train_new_model(df):
    # Load the pickled model and scaler
    with open("model/best_svm.pkl", "rb") as f:
        loaded_model = pickle.load(f)

    with open("model/scaler.pkl", "rb") as f:
        loaded_scaler = pickle.load(f)

    # Scale the new data
    new_X = df.drop(columns=["Price Direction"])

    # Reorder the columns in the new data to match the original data
    column_order = [
        "unix",
        "open",
        "high",
        "low",
        "close",
        "Volume BTC",
        "Volume USD",
        "timestamp",
        "7-day SMA",
        "21-day SMA",
        "EMA_0.67",
        "12-day EMA",
        "26-day EMA",
        "MACD",
        "20-day STD",
        "Upper BB",
        "Lower BB",
        "High-Low Spread",
        "MA Indicator",
        "Price Diff",
    ]
    new_X = new_X[column_order]

    # Scale the new data
    new_X_scaled = loaded_scaler.transform(new_X)

    # Make predictions on the new data
    predictions = loaded_model.predict(new_X_scaled)

    return predictions
