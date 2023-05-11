import unittest

import pandas as pd
import tweepy

from get_tweets import get_tweet_data, get_tweets
class ProgramTest(unittest.TestCase):

    def test_get_tweets(self):
        # Test getting tweets
        tweets = get_tweets()
        self.assertGreaterEqual(len(tweets), 0)
        for tweet in tweets:
            self.assertTrue(tweet.favorite_count >= 5 or tweet.retweet_count >= 2)

    def test_get_tweet_data(self):
        # Test getting tweet data
        tweets = [
            tweepy.Status(
                id=1,
                created_at=pd.Timestamp("2022-01-01 12:00:00"),
                full_text="This is a tweet about bitcoin.",
                favorite_count=5,
                retweet_count=2,
            ),
            tweepy.Status(
                id=2,
                created_at=pd.Timestamp("2022-01-01 13:00:00"),
                full_text="This is another tweet about bitcoin.",
                favorite_count=3,
                retweet_count=1,
            ),
        ]
        df = get_tweet_data(tweets)
        self.assertEqual(len(df), 2)
        self.assertIn("date", df.columns)
        self.assertIn("text", df.columns)

if __name__ == '__main__':
    unittest.main()
