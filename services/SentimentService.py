from services.NLPService import clean, extractOrg
from services.DBService import searchTweetByOrg, searchTweetByMultipleOrgs

def getSentimentAnalysis(tweet):
    orgList = extractOrg(clean(tweet))
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
        # result = getSentiment(tweet)
        result.append(5)
    return result

# receive list of sentiment score
def average(score):
    sum = 0.0
    for i in score:
        sum += i
    return sum / len(score)