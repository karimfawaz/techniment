import tweepy

# Authentication credentials
consumer_key = 'Wv0nW3p23AX5EV3Tsh5gmURNx'
consumer_secret = 'Hcy5AH41z6Sclh9AwdHFoJta679VCnMzoOrFawYEkAejkQ40DN'
access_token = '1734435330-yATHuq6RVia2N6IDE9Bql4qn0StZJqqT9ul8Vsy'
access_token_secret = 'jUcHGRjz1zVYzfYlC0BcmoL5Ef6dCegIhjSqk5ETlthqF'

# Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

client = tweepy.API(auth)

# Pull tweets from twitter
query = 'BTC'
paginator = tweepy.Paginator(
    client.search_recent_tweets,   # method we want to use
    query=query,                   # argument for this method
    max_results=100,               # how many tweets per page
    limit=10                       # how many pages to retrieve
)
# Print the pulled tweets
tweet_list = []
for tweet in paginator.flatten():  # Total number of tweets to retrieve
    tweet_list.append(tweet)
    print(tweet)

# # Fetch tweets
# tweets = client.search_tweets(q='#BTC', count=100)

# # Print tweets
# for tweet in tweets:
#     print(tweet.text)
