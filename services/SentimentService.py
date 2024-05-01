from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import *
import pickle
import json

from services.NLPService import clean, extractOrg, extractProductOllama
from services.DBService import searchTweetByMultipleOrgs, listToDictEmptyArray, searchTweetContainProductAndOrg


def getSentimentAnalysis(tweet):
    response = extractProductOllama(tweet)
    responseJSONString = str(response['message']['content']).replace('\'None\'', '').replace('\'', '\"')
    responseJSON = json.loads(responseJSONString)

    try:
        orgList = responseJSON['ORG']
    except:
        orgList = []

    if orgList == None:
        return

    orgsWithTweets = searchTweetByMultipleOrgs(orgList)

    result = []
    for org in orgsWithTweets:
        data = {}
        sentiments = analyze(orgsWithTweets[org])
        mean = fuzzyLogic(average(sentiments))
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
    with open('resources/model.pkl', 'rb') as f:
        clf2 = pickle.load(f)

    text = clean(text)
    data = clf2.predict([text])
    return data[0]


def fuzzyLogic(sentiments):
    fuzzy = {
        'angry': 0,
        'sad': 0,
        'neutral': 0,
        'happy': 0,
        'excited': 0
    }
    # < -6 Angry
    if sentiments <= -6:
        fuzzy['angry'] += 1
    # -5 - -2 Sad
    if sentiments > -5 and sentiments < -2:
        fuzzy['sad'] += sentiments + 5 / 3

    if sentiments == -2:
        fuzzy['sad'] += 1

    if sentiments > -2 and sentiments < 0:
        fuzzy['sad'] += abs(sentiments) / 2

    # 0 Neutral

    if sentiments > -1 and sentiments < 0:
        fuzzy['neutral'] += abs(sentiments - 1)

    if sentiments == 0:
        fuzzy['neutral'] += 1

    if sentiments > 0 and sentiments < 1:
        fuzzy['neutral'] += sentiments
    # 0 - 3 Happy
    if sentiments > 0 and sentiments < 2:
        fuzzy['happy'] += sentiments / 2

    if sentiments == 2:
        fuzzy['happy'] += 1

    if sentiments > 2 and sentiments < 5:
        fuzzy['happy'] += sentiments / 5
    # > 3 Excited
    if sentiments > 5 and sentiments < 6:
        fuzzy['excited'] += sentiments - 5

    response = ""
    i = 0
    for key in fuzzy:
        if fuzzy[key] > 0:
            if i != 0:
                response += ", "
            response += key + ": " + f'{fuzzy[key]:.2f}'
            i += 1


    return response
