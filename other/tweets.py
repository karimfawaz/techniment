import tweepy
import pandas as pd
import datetime


def get_df():
    return pd.DataFrame(
        columns=[
            "tweet_id",
            "name",
            "screen_name",
            "retweet_count",
            "text",
            "mined_at",
            "created_at",
            "favourite_count",
            "hashtags",
            "status_count",
            "followers_count",
            "location",
            "source_device",
        ]
    )
    

class TweetMiner(object):

    result_limit = 20
    data = []
    api = False

    twitter_keys = {
        "consumer_key": "cLwV1vwnMhb4xFxu4lPYXQGoC",
        "consumer_secret": "YpYWFyVSdPCSkU1n48iELSbU64D1nqGLNhkuFXV58f0ngN6ALQ",
        # "access_token_key": "<To be replace>",
        # "access_token_secret": "<To be replace>",
    }

    def __init__(self, keys_dict=twitter_keys, api=api):

        self.twitter_keys = keys_dict

        auth = tweepy.OAuthHandler(
            keys_dict["consumer_key"], keys_dict["consumer_secret"]
        )
        # auth.set_access_token(
        #     keys_dict["access_token_key"], keys_dict["access_token_secret"]
        # )

        self.api = tweepy.API(
            auth, wait_on_rate_limit=True
        )
        self.twitter_keys = keys_dict

    def mine_crypto_currency_tweets(self, query="BTC"):

        last_tweet_id = False
        page_num = 1

        data = get_df()
        cypto_query = f"#{query}"
        print(" ===== ", query, cypto_query)
        for page in tweepy.Cursor(
            self.api.search_tweets,
            q=cypto_query,
            lang="en",
            tweet_mode="extended",
            count=200,  # max_id=1295144957439690000
        ).pages():
            print(" ...... new page", page_num)
            page_num += 1

            for item in page:
                mined = {
                    "tweet_id": item.id,
                    "name": item.user.name,
                    "screen_name": item.user.screen_name,
                    "retweet_count": item.retweet_count,
                    "text": item.full_text,
                    "mined_at": datetime.datetime.now(),
                    "created_at": item.created_at,
                    "favourite_count": item.favorite_count,
                    "hashtags": item.entities["hashtags"],
                    "status_count": item.user.statuses_count,
                    "followers_count": item.user.followers_count,
                    "location": item.place,
                    "source_device": item.source,
                }

                try:
                    mined["retweet_text"] = item.retweeted_status.full_text
                except:
                    mined["retweet_text"] = "None"

                last_tweet_id = item.id
                data = data.append(mined, ignore_index=True)

            if page_num % 180 == 0:
                date_label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                print("....... outputting to csv", page_num, len(data))
                data.to_csv(
                    f"{query}_{page_num}_{date_label}.csv", index=False)
                print("  ..... resetting df")
                data = get_df()

        date_label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        data.to_csv(f"{query}_{page_num}_{date_label}.csv", index=False)


miner = TweetMiner()
miner.mine_crypto_currency_tweets()
handle_list = [
    "BTC",
    "ETH",
    "USDT",
    "XRP",
    "BCH",
    "ADA",
    "BSV",
    "LTC",
    "LINK",
    "BNB",
    "EOS",
    "TRON",
]