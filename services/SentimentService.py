from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import *
import pickle

from services.NLPService import clean, extractProduct, extractOrg
from services.DBService import searchTweetByProduct, listToDictEmptyArray, searchTweetContainProductAndOrg

def getSentimentAnalysis(tweet):
    productName = 'aws'
    # productName = extractProduct(clean(tweet))
    if productName == None:
        return

    productTweets = searchTweetByProduct(productName)
    orgs = findOrgInTweets(productTweets)
    orgsWithTweets = {}
    for org in orgs:
        tweets = searchTweetContainProductAndOrg(productName, org)
        if len(tweets) > 0:
            orgsWithTweets[org] = tweets

    result = []
    for org in orgsWithTweets:
        data = {}
        sentiments = analyze(orgsWithTweets[org])
        mean = average(sentiments)
        data['brand'] = org
        data['sentiment'] = mean
        data['examples'] = sentiments[:2]
        result.append(data)
    return result

# receive list of tweets from an org
def analyze(tweets):
    result = []
    for i in range(len(tweets)):
        # do sentiment analysis
        data = {}
        score = predictData(tweets[i]['text'])
        data['sentiment'] = int(score)
        data['text'] = tweets[i]['text']
        data['from'] = tweets[i]['userId']
        result.append(data)
    return result

def findOrgInTweets(tweets):
    orgSet = set()
    for tweet in tweets:
        org = extractOrg(clean(tweet))
        if org == None:
            continue
        orgSet.add(org)
    dict = listToDictEmptyArray(list(orgSet))
    return dict

# receive list of sentiment score
def average(sentiments):
    sum = 0.0
    for i in range(len(sentiments)):
        sum += sentiments[i]['sentiment']
    return sum / len(sentiments)

def predictData(text):
    with open('resources/model.pkl','rb') as f:
        clf2 = pickle.load(f)

    text = clean(text)
    data = clf2.predict([text])
    return data[0]