import tweepy

# Put your Bearer Token in the parenthesis below
auth = tweepy.OAuthHandler('Wv0nW3p23AX5EV3Tsh5gmURNx',
                           'Hcy5AH41z6Sclh9AwdHFoJta679VCnMzoOrFawYEkAejkQ40DN')
auth.set_access_token('1734435330-yATHuq6RVia2N6IDE9Bql4qn0StZJqqT9ul8Vsy',
                      'jUcHGRjz1zVYzfYlC0BcmoL5Ef6dCegIhjSqk5ETlthqF')
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
