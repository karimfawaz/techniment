"""

A PROGRAM THAT FETCHES 5 RELEVANT BITCOIN TWEETS PER HOUR FOR THE PAST 2000 HOURS FROM TWITTER API WITH ELEVATED ACCESS

"""
import tweepy
import pandas as pd

API_KEY = "MuahFm88RanZjtqzqmn6ZYCyN"
API_SECRET_KEY = "dl4oRLMvIBZ5pHGBC9dIBZgrL5f8iiodJ083nmGMkTeTAGDAum"
ACCESS_TOKEN = "1734435330-ifCchqPOechU7q7HjFH26rOcDsyv5BRTr2AY78C"
ACCESS_TOKEN_SECRET = "RHpAXJotGyBLe0JgXzDVr1RMPnqeLhI7qEAvSSEfC7YLP"


def get_tweets():
    auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    tweets = []
    for tweet in tweepy.Cursor(
        api.search_tweets,
        q="bitcoin -filter:retweets",
        lang="en",
        tweet_mode="extended",
    ).items(1000):
        if tweet.favorite_count >= 5 or tweet.retweet_count >= 2:
            tweets.append(tweet)
    return tweets


def get_tweet_data(tweets):
    tweet_data = [[tweet.created_at, tweet.full_text] for tweet in tweets]
    df = pd.DataFrame(tweet_data, columns=["date", "tweet"])
    return df


tweets = get_tweets()
df = get_tweet_data(tweets)
print(df)
df.to_csv("./data/tweets.csv", index=False)
