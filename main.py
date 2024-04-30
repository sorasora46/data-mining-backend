from fastapi import FastAPI
from models.TweetRequest import TweetRequest
from services.SentimentService import getSentimentAnalysis

app = FastAPI()

@app.get("/sentiment")
def getSentimentsFromTweet(tweetRequest: TweetRequest):
    result = getSentimentAnalysis(tweetRequest.tweet)
    return { "result": result }

"""
[x] โหลด tweet จาก url
[x] เอา text มาทำ ner (ใช้ model จาก spacy)
[x] เอา ner มา search tweet ตามแบรน (แบรนละ 100 tweet)
[] (sentiment analysis) เอาแต่ละ tweet ของแต่ละแบรนมาทำ sentiment analysis
    [] find sentiment for each tweet in each brand (org)
    [] เฉลี่ย sentiment ของแต่ละแบรน
[] return sentiment analysis result to user
"""