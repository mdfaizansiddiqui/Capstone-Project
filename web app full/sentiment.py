import os
import pandas as pd
import re
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def cleaner(tweet):
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    tweet = tweet.replace("\n", "")
    return tweet


def sentiment_scores(sentence):
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
    # polarity_scores method of SentimentIntensityAnalyzer
    # oject gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)
    return sentiment_dict['compound']

def get_sentiment(start_date, end_date, stock_name):
    key_word = stock_name +" stock"
    cmd = "snscrape --jsonl --since "+ start_date + " twitter-search '" + key_word + " until:" + end_date +"' > tweets.json"
    os.system(cmd)
    tweets_df = pd.read_json('tweets.json', lines=True)
    if len(tweets_df) in range(2):
       cmd = "snscrape --jsonl --since "+ start_date + " twitter-search '" + stock_name + " until:" + end_date +"' > tweets.json"
       os.system(cmd)
       tweets_df = pd.read_json('tweets.json', lines=True)
    if len(tweets_df) < 2:
        tweets_df = pd.read_json('tweets1.json', lines=True)
    final_df = tweets_df.sort_values(by=['retweetCount', 'replyCount'], ascending=False)
    final_df['Cleaned Tweets'] = final_df['renderedContent'].apply(cleaner)
    min_tweets = min(15, len(final_df))  
    tweets = np.array(final_df['Cleaned Tweets'][:min_tweets])
    sentiment_score_list = []
    for tweet in tweets:
        sentiment = sentiment_scores(tweet)
        sentiment_score_list.append(sentiment)
    return np.average(sentiment_score_list)

print(get_sentiment('2021-01-15', '2021-01-16', 'amazon'))