import unittest
import pandas as pd
import numpy as np
from dateutil.parser import parse
from multimodal_model import (
    create_multimodal_model,
    is_valid_date,
    get_filtered_tweets,
    get_hourly_sentiment,
    get_new_data,
)

class TestMultimodalModel(unittest.TestCase):
    def test_is_valid_date(self):
        self.assertTrue(is_valid_date("2021-05-09"))
        self.assertTrue(is_valid_date("2021-05-09T12:30:01"))
        self.assertFalse(is_valid_date("Not a date"))
        self.assertFalse(is_valid_date(123456))
        self.assertFalse(is_valid_date(""))

    def test_get_filtered_tweets(self):

        get_filtered_tweets()
        tweets = pd.read_csv("./data/2022_tweets.csv")
        self.assertTrue(len(tweets) > 0)
        self.assertTrue("date" in tweets.columns)
        self.assertTrue("text" in tweets.columns)
        self.assertTrue("date_hour" in tweets.columns)

    def test_get_hourly_sentiment(self):

        hourly_sentiment_df = get_hourly_sentiment()
        self.assertTrue(len(hourly_sentiment_df) > 0)
        self.assertTrue("date_hour" in hourly_sentiment_df.columns)
        self.assertTrue("sentiment" in hourly_sentiment_df.columns)

    def test_get_new_data(self):
        new_data = get_new_data()
        self.assertTrue(len(new_data) > 0)
        self.assertTrue("timestamp" in new_data.columns)
        self.assertTrue("7-day SMA" in new_data.columns)
        self.assertTrue("21-day SMA" in new_data.columns)
        self.assertTrue("EMA_0.67" in new_data.columns)
        self.assertTrue("12-day EMA" in new_data.columns)
        self.assertTrue("26-day EMA" in new_data.columns)
        self.assertTrue("MACD" in new_data.columns)
        self.assertTrue("20-day STD" in new_data.columns)
        self.assertTrue("Upper BB" in new_data.columns)
        self.assertTrue("Lower BB" in new_data.columns)
        self.assertTrue("High-Low Spread" in new_data.columns)
        self.assertTrue("MA Indicator" in new_data.columns)
        self.assertTrue("sentiment" in new_data.columns)
        self.assertTrue("Price Diff" in new_data.columns)

if __name__ == "__main__":
    unittest.main()
