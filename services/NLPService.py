import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stopWords = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))

# 'en_core_web_trf' -> for accuracy
# 'en_core_web_sm' -> for efficient
spacyModel = 'en_core_web_trf'

nlp = spacy.load(spacyModel)

def extractOrg(text):
    doc = nlp(text)
    orgEnts = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
    if len(orgEnts) > 0:
        return orgEnts
    return text

def extractLink(text):
    doc = nlp(text)
    urls = [token.text for token in doc if token.like_url]
    if len(urls) > 0:
        return urls
    return

def extractUsername(text):
    texts = text.split(' ')
    users = [string for string in texts if string.startswith('@')]
    if len(users) > 0:
        return users
    return
    
def clean(text):
    urls = extractLink(text)
    users = extractUsername(text)

    if urls is not None:
        for url in urls:
            text = text.replace(url, '')

    if users is not None:
        for user in users:
            text = text.replace(user, '')

    text = text.replace('rt', '')
    text = text.replace('RT', '')
    text = text.replace('Rt', '')
    text = text.replace('rR', '')
    
    text = text.replace('#', '')

    tokens = text.split(' ')
    tokens = [token for token in tokens if token.lower() not in stopWords]

    
    cleanedText = ' '.join(tokens)

    return cleanedText