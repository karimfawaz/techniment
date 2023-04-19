import tweepy
import re
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timedelta
import pytz
import time

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


# Define a function to collect tweets
def collect_tweets(keywords, count=100, lang='en', since_id=None, min_retweets=10, min_favorites=10, hours=1):
    tweets = []
    utc = pytz.UTC
    min_date = datetime.now(utc) - timedelta(hours=hours)
    for tweet in tweepy.Cursor(api.search_tweets, q=keywords, lang=lang, since_id=since_id, tweet_mode='extended').items(count):
        if tweet.created_at > min_date and tweet.retweet_count >= min_retweets and tweet.favorite_count >= min_favorites:
            tweets.append(tweet.full_text)
    return tweets


# Define a function to preprocess the collected tweets
def preprocess_tweets(tweets):
    preprocessed_tweets = []
    for tweet in tweets:
        tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet,
                       flags=re.MULTILINE)  # Remove URLs
        tweet = re.sub(r'\@\w+|\#', '', tweet)  # Remove mentions and hashtags
        preprocessed_tweets.append(tweet)
    return preprocessed_tweets


# Load the CryptoBERT model
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
sentiment_pipeline = pipeline(
    "text-classification", tokenizer=tokenizer, model=model)

# Define a function to predict sentiment scores using the CryptoBERT model


def predict_sentiments(tweets):
    sentiments = sentiment_pipeline(tweets)
    return sentiments


# Collect and preprocess tweets
keywords = 'bitcoin OR btc'
tweets = collect_tweets(keywords, min_retweets=3,
                        min_favorites=0, hours=1, count=100)
print(tweets)
preprocessed_tweets = preprocess_tweets(tweets)
print(preprocessed_tweets)
# Predict sentiment scores
sentiments = predict_sentiments(preprocessed_tweets)


# Aggregate sentiment scores
sentiment_df = pd.DataFrame(sentiments).apply(pd.to_numeric, errors='coerce')
sentiment_df['label'] = sentiment_df.idxmax(axis=1)
sentiment_counts = sentiment_df['label'].value_counts(normalize=True)

print(sentiment_counts)


client = tweepy.Client(
    bearertoken, wait_on_rate_limit=True)


def scrape_mentions(FILE_NAME, query_term_liste):
    start = time.time()
    df_counter = 0

    for i in query_term_liste:

        tweets = []
        counter = 0

        for response in tweepy.Paginator(client.search_all_tweets,
                                         query=f'({i}) lang:de',
                                         user_fields=[
                                             'username', 'public_metrics', 'description', 'location'],
                                         tweet_fields=[
                                             'created_at', 'geo', 'public_metrics', 'text'],
                                         expansions=[
                                             'author_id', 'entities.mentions.username'],
                                         start_time='2022-02-01T00:00:00Z',
                                         end_time='2022-04-01T00:00:00Z',
                                         max_results=500):
            time.sleep(1)
            tweets.append(response)
            counter = counter + 1
            print(f"Response Nummer: {counter}")

        end = time.time()
        print(f"Das Scrapen von hat {(end - start)/60} Minuten gebraucht.")
        print(len(tweets))

        result = []
        user_dict = {}
        # Loop through each response object
        for response in tweets:
            # Take all of the users, and put them into a dictionary of dictionaries with the info we want to keep
            for user in response.includes['users']:
                user_dict[user.id] = {'username': user.username,
                                      'followers': user.public_metrics['followers_count'],
                                      'tweets': user.public_metrics['tweet_count'],
                                      'description': user.description,
                                      'location': user.location
                                      }
            for tweet in response.data:
                # print(tweet.entities["mentions"])
                if tweet.entities != None:
                    mentions = tweet.entities["mentions"]
                else:
                    mentions = "None"

                mention_in_tweet = []

                for d in mentions:
                    username = d
                    if isinstance(username, dict):
                        mention_in_tweet.append(username["username"])
                    else:
                        username = "NONE"
                        mention_in_tweet.append(username)
                        # print(mention_in_tweet)
                    # For each tweet, find the author's information
                author_info = user_dict[tweet.author_id]
                # Put all of the information we want to keep in a single dictionary for each tweet
                result.append({'author_id': tweet.author_id,
                               'username': author_info['username'],
                               'author_followers': author_info['followers'],
                               'author_tweets': author_info['tweets'],
                               'author_description': author_info['description'],
                               'author_location': author_info['location'],
                               'text': tweet.text,
                               'created_at': tweet.created_at,
                               'quote_count': tweet.public_metrics['quote_count'],
                               'retweets': tweet.public_metrics['retweet_count'],
                               'replies': tweet.public_metrics['reply_count'],
                               'likes': tweet.public_metrics['like_count'],
                               'mentioned': mention_in_tweet
                               })

            # Change this list of dictionaries into a dataframe
            df = pd.DataFrame(result)
            df.sort_values(by=['created_at'], ascending=False)
            print(
                "DIESER RUN IST FERTIG**************************************************")

        df_counter = df_counter + 1

        df["Run"] = f"{FILE_NAME}_polis_{df_counter}"
        df.to_csv(f"mentions_{FILE_NAME}_{df_counter}.csv")
        print(len(df))
