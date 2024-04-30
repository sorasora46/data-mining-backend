from services.NLPService import clean, extractOrg
from services.DBService import searchTweetByOrg, searchTweetByMultipleOrgs

def getSentimentAnalysis(tweet):
    orgList = extractOrg(clean(tweet))
    orgTweets = searchTweetByMultipleOrgs(orgList)
    orgSentiment = orgTweets
    for org in orgTweets:
        print(orgTweets[org])
        # orgSentiment[org] = average(analyze(orgTweets[org]))
    return orgSentiment

# receive list of tweets from an org
def analyze(tweets):
    result = []
    for i in len(tweets):
        # do sentiment analysis
        # result = getSentiment(tweet)
        result[i] = 5
        break
    return result

def average(score):
    sum = 0.0
    print(score)
    return 0
    # for i in score:
    #     sum += i
    # return sum / len(score)