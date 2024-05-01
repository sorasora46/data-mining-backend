from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import *
import pickle

from services.NLPService import clean, extractProduct
from services.DBService import searchTweetByMultipleOrgs

def getSentimentAnalysis(tweet):
    productList = extractProduct(clean(tweet))
    if productList == None:
        return

    orgTweets = searchTweetByMultipleOrgs(orgList)
    orgSentiment = orgTweets
    for org in orgTweets:
        orgSentiment[org] = average(analyze(orgTweets[org]))
    return orgSentiment

# receive list of tweets from an org
def analyze(tweets):
    result = []
    for i in range(len(tweets)):
        # do sentiment analysis
        score = predictData(tweets[i])
        result.append(score)
    return result

# receive list of sentiment score
def average(score):
    sum = 0.0
    for i in score:
        sum += i
    return sum / len(score)

def predictData(text):
    with open('resources/model.pkl','rb') as f:
        clf2 = pickle.load(f)
    
    cv=CountVectorizer(max_features=5000,stop_words='english')
    text = clean(text)
    data = clf2.predict([text])
    return data[0]

# def stem(text):
#     ps=PorterStemmer()
#     y=[]
#     for i in text.split():
#         y.append(ps.stem(i))
#     return " ".join(y)

# def preprocessing(text):
#     # lower case
#     text = text.lower()
#     # remove url
#     text = ' '.join(x for x in text.split() if 
#                     not x.startswith("http://") or 
#                     not x.startswith("https://")
#                     )
#     text = ' '.join(x for x in text.split() if 
#                     not x.startswith("@")
#                     )
#     text = text.replace("rt","")
#     # trim
#     text = text.strip()
#     # remove stop words
#     text = stem(text)
#     return text