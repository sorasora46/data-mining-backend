from pydantic import BaseModel

class TweetRequest(BaseModel):
    tweet: str