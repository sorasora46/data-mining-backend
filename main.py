from fastapi import FastAPI
from models.TweetRequest import TweetRequest
from services.SentimentService import getSentimentAnalysis
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/sentiment")
def getSentimentsFromTweet(tweetRequest: TweetRequest):
    result = getSentimentAnalysis(tweetRequest.tweet)
    return { "result": result }

"""
[x] โหลด tweet จาก url
[x] เอา text มาทำ ner (ใช้ model จาก spacy)
[x] เอา ner มา search tweet ตามแบรน (แบรนละ 100 tweet)
[x] (sentiment analysis) เอาแต่ละ tweet ของแต่ละแบรนมาทำ sentiment analysis
    [x] find sentiment for each tweet in each brand (org)
    [x] เฉลี่ย sentiment ของแต่ละแบรน
[x] return sentiment analysis result to user
[] apply real sentiment model & fuzzy logic
"""