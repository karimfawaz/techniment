from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)
from dateutil.parser import parse
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import requests
import re

from model.get_tweets import save_tweets


def create_multimodal_model():
    url = "./data/Bitstamp_BTCUSD_1h.csv"
    data = pd.read_csv(url)

    # Convert the date column to a Unix timestamp
    data["timestamp"] = pd.to_datetime(data["date"]).astype(np.int64) // 10**9

    # Remove the date and symbol columns
    data = data.drop(columns=["date", "symbol"])

    # Filter the technical data to include only data from 2022
    start_date = int(pd.Timestamp("2022-01-01").timestamp())
    end_date = int(pd.Timestamp("2022-12-31").timestamp())

    data = data[(data["timestamp"] >= start_date) & (data["timestamp"] <= end_date)]
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

    hourly_sentiment_df = pd.read_csv("./data/hourly_sentiment.csv")
    hourly_sentiment_df["date_hour"] = pd.to_datetime(hourly_sentiment_df["date_hour"])
    hourly_sentiment_df["date_hour"] = (
        pd.to_datetime(hourly_sentiment_df["date_hour"]).astype(np.int64) // 10**9
    )

    data = data.merge(
        hourly_sentiment_df, left_on="timestamp", right_on="date_hour", how="left"
    )
    data["sentiment"].fillna(0, inplace=True)
    data = data.drop(columns=["date_hour"])

    # Calculate the price difference between consecutive periods
    data["Price Diff"] = data["close"].diff()

    # Create a binary variable that indicates whether the price has gone up (1) or down (0)
    data["Price Direction"] = np.where(data["Price Diff"] > 0, 1, 0)

    # Drop rows with NaN values
    data = data.dropna()
    # Sort date from oldest to newest unix timestamp
    data = data.sort_values("timestamp", ascending=True)
    data.to_csv("./data/processed_data.csv", index=False)
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

    print("Creating the Scaler")
    # Apply feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Define the parameter grid for GridSearchCV
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
    }
    print("Fitting using the best params")

    # Use GridSearchCV to find the best hyperparameters
    grid = GridSearchCV(LinearSVC(), param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train_scaled, y_train)

    print("Best parameters found: ", grid.best_params_)

    # Train the model with the best hyperparameters
    best_svm = LinearSVC(C=grid.best_params_["C"])
    best_svm.fit(X_train_scaled, y_train)

    print("Predicting using the model")

    # Make predictions on the validation set
    y_val_pred = best_svm.predict(X_val_scaled)

    # Calculate the accuracy and print the classification report for the validation set
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(classification_report(y_val, y_val_pred))

    # Make predictions on the test set
    y_test_pred = best_svm.predict(X_test_scaled)

    # Calculate the accuracy and print the classification report for the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(classification_report(y_test, y_test_pred))
    def calculate_profit(y_test, y_test_pred, close_prices):
        initial_capital = 0
        btc_holdings = 0
        short_positions = 0

        for i in range(len(y_test_pred) - 1):
            if y_test_pred[i] == 1:
                if short_positions > 0:
                    # Close short position
                    initial_capital -= close_prices[i]
                    short_positions -= 1
                else:
                    # Open long position
                    btc_holdings += 1
                    initial_capital -= close_prices[i]
            elif y_test_pred[i] == 0:
                if btc_holdings > 0:
                    # Close long position
                    btc_holdings -= 1
                    initial_capital += close_prices[i]
                else:
                    # Open short position
                    short_positions += 1
                    initial_capital += close_prices[i]

        # Close remaining positions at the end
        initial_capital += btc_holdings * close_prices[-1] - short_positions * close_prices[-1]

        return initial_capital

    # Calculate the profit made using the model's predictions
    model_profit = calculate_profit(y_test, y_test_pred, X_test["close"].values)
    print(f"Profit made using the model's predictions: ${model_profit:.2f}")

    # Calculate the profit made using the buy and hold approach
    buy_and_hold_profit = X_test["close"].values[-1] - X_test["close"].values[0]
    print(f"Profit made using the buy and hold approach: ${buy_and_hold_profit:.2f}")

    # Compare the model's profit to the buy and hold profit
    if model_profit > buy_and_hold_profit:
        print("The model's trading strategy outperformed the buy and hold approach.")
    elif model_profit < buy_and_hold_profit:
        print("The buy and hold approach outperformed the model's trading strategy.")
    else:
        print("The model's trading strategy and the buy and hold approach performed equally.")

    print("Saving the model")

    # Save the trained model and scaler to disk
    with open("model/multimodal_model.pkl", "wb") as file:
        pickle.dump(best_svm, file)
    with open("model/multimodal_scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

def train_new_model(df):
    # Load the pickled model and scaler
    with open("model/multimodal_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)

    with open("model/multimodal_scaler.pkl", "rb") as f:
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
        "sentiment",
        "Price Diff",
    ]
    new_X = new_X[column_order]

    # Scale the new data
    new_X_scaled = loaded_scaler.transform(new_X)

    # Make predictions on the new data
    predictions = loaded_model.predict(new_X_scaled)

    return predictions


# create_multimodal_model()


def is_valid_date(date_string):
    if not isinstance(date_string, str) or len(date_string) < 6:
        return False
    try:
        parse(date_string)
        return True
    except (ValueError, TypeError, OverflowError):
        return False


def get_filtered_tweets():
    # Filter the technical data to include only data from 2022
    start_date = pd.Timestamp("2022-01-01")
    end_date = pd.Timestamp("2022-12-31")

    tweets = pd.read_csv("./data/Bitcoin_tweets.csv")

    # Filter out rows with invalid date strings
    valid_dates_mask = tweets["date"].apply(is_valid_date)
    tweets = tweets[valid_dates_mask]

    # Now, you can safely convert the 'date' column to datetime objects
    tweets["date"] = pd.to_datetime(tweets["date"])

    # Filter the tweets data to include only data from 2022
    tweets["date"] = pd.to_datetime(tweets["date"])
    tweets = tweets[(tweets["date"] >= start_date) & (tweets["date"] <= end_date)]

    # Remove unnecessary columns from the tweets data
    tweets = tweets[["date", "text"]]

    # Remove duplicate rows based on the 'text' column
    tweets.drop_duplicates(subset="text", inplace=True)

    def is_ad_or_scam(text):
        ad_keywords = [
            "promo",
            "giveaway",
            "free",
            "discount",
            "limited time",
            "offer",
            "bonus",
            "buy now",
            "sale",
        ]
        scam_keywords = [
            "double your bitcoin",
            "send me and I will return",
            "guaranteed profit",
            "no risk",
        ]
        bot_keywords = ["bot", "automated", "robot", "algorithm"]

        keywords = ad_keywords + scam_keywords + bot_keywords

        for keyword in keywords:
            if keyword.lower() in text.lower():
                return True

        return False

    def remove_links_mentions_hashtags(text):
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)

        return text.strip()

    def preprocess_tweet(text):
        text = remove_links_mentions_hashtags(text)

        if is_ad_or_scam(text):
            return None
        else:
            return text

    # Apply the preprocessing function to the 'text' column
    tweets["text"] = tweets["text"].apply(preprocess_tweet)

    # Remove rows with None in the 'text' column (i.e., filtered-out tweets)
    tweets = tweets[tweets["text"].notnull()]

    # Round up the 'date' column to the nearest hour
    tweets["date_hour"] = tweets["date"].dt.ceil("H")

    # Limit each hour to have 5 tweets
    tweets = tweets.groupby("date_hour").head(5).reset_index(drop=True)

    # Sort the tweets by date
    tweets.sort_values(by="date", inplace=True)

    # Save the filtered tweets to a CSV file
    tweets.to_csv("./data/2022_tweets.csv", index=False)


# print("Getting filtered tweets...")
# get_filtered_tweets()
# print("Done!")


def get_hourly_sentiment(tweets="./data/2022_tweets.csv"):
    # Load the preprocessed 2022 tweets
    tweets = pd.read_csv(tweets)

    # Filter out rows with invalid date strings
    tweets = tweets[tweets["date"].apply(is_valid_date)]

    # Now, you can safely convert the 'date' column to datetime objects
    tweets["date"] = pd.to_datetime(tweets["date"])

    # Load CryptoBERT model and tokenizer
    model_name = "ElKulako/cryptobert"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Create the sentiment analysis pipeline
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        max_length=64,
        truncation=True,
        padding="max_length",
    )

    def label_to_number(label):
        if label == "Bearish":
            return -1
        elif label == "Neutral":
            return 0
        elif label == "Bullish":
            return 1

    def calculate_sentiment(text):
        preds = pipe([text])
        sentiment_label = preds[0]["label"]
        sentiment_score = preds[0]["score"]
        sentiment_number = label_to_number(sentiment_label)
        return sentiment_number * sentiment_score

    tweets["sentiment"] = tweets["text"].apply(calculate_sentiment)

    # Group by hourly intervals and aggregate the sentiment scores
    tweets["date_hour"] = tweets["date"].dt.floor("H")
    hourly_sentiment = tweets.groupby("date_hour")["sentiment"].mean()

    # Convert the aggregated sentiment scores to a pandas DataFrame
    hourly_sentiment_df = pd.DataFrame(hourly_sentiment).reset_index()

    return hourly_sentiment_df

    # # Analyze sentiment of the tweets
    # tweets['sentiment'] = tweets['text'].apply(lambda x: pipe(x)[0]['label'])

    # # Group by hourly intervals and aggregate the sentiment scores
    # tweets['date_hour'] = tweets['date'].dt.floor('H')
    # hourly_sentiment = tweets.groupby('date_hour')['sentiment'].agg(
    #     lambda x: x.value_counts().index[0])


# hourly_sentiment_df = get_hourly_sentiment()

# print(hourly_sentiment_df)
# hourly_sentiment_df.to_csv("./data/hourly_sentiment.csv", index=False)

def get_new_data():
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
    # save_tweets()
    tweets_sentiment = get_hourly_sentiment("./data/tweets.csv")
    tweets_sentiment['date_hour'] = pd.to_datetime(tweets_sentiment['date_hour']).astype(np.int64) // 10**9

    df = df.merge(
        tweets_sentiment, left_on="timestamp", right_on="date_hour", how="left"
    )
    df["sentiment"].fillna(0, inplace=True)
    df = df.drop(columns=["date_hour"])

    return df

# get_new_data()