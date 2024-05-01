import pandas as pd

dataset = pd.read_csv('resources/tweets-engagement-metrics.csv')

def searchTweetByProduct(text, n = 50):
    targetDataFrame = dataset[dataset['text'].str.contains(text, case=False)]
    firstNRows = targetDataFrame.head(n)
    results = firstNRows['text'].tolist()
    resultsNoDuplicate = set(results)
    return list(resultsNoDuplicate)

def searchTweetContainProductAndOrg(product, org, n = 50):
    targetDataFrame = dataset[dataset['text'].str.contains(product, case=False) & dataset['text'].str.contains(org, case=False)]
    firstNRows = targetDataFrame.head(n)

    unique_tweets = set()  # Set to track unique tweets
    results = []
    for _, row in firstNRows.iterrows():
        tweet_text = row['text']
        # Check if the tweet text is already in the set of unique tweets
        if tweet_text not in unique_tweets:
            tweet_dict = {'userId': row['UserID'], 'text': tweet_text}
            results.append(tweet_dict)
            unique_tweets.add(tweet_text)  # Add the tweet text to the set of unique tweets
    return results

def searchTweetByOrg(text, n = 50):
    targetDataFrame = dataset[dataset['text'].str.contains(text, case=False)]
    firstNRows = targetDataFrame.head(n)

    unique_tweets = set()  # Set to track unique tweets
    results = []
    for _, row in firstNRows.iterrows():
        tweet_text = row['text']
        # Check if the tweet text is already in the set of unique tweets
        if tweet_text not in unique_tweets:
            tweet_dict = {'userId': row['UserID'], 'text': tweet_text}
            results.append(tweet_dict)
            unique_tweets.add(tweet_text)  # Add the tweet text to the set of unique tweets
    return results

def searchTweetByMultipleOrgs(orgs, n = 50):
    result = listToDictEmptyArray(orgs)
    for org in orgs:
        result[org] = searchTweetByOrg(org, n)
    return result
    
def listToDictEmptyArray(list):
    dict = {}
    for i in range(len(list)):
        dict[list[i]] = []
    return dict
