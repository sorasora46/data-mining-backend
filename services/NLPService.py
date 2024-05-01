import spacy
import requests
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stopWords = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))

# 'en_core_web_trf' -> for accuracy
# 'en_core_web_sm' -> for efficient
spacyModel = 'en_core_web_sm'
# spacyModel = 'en_core_web_trf'
# nerModel = 'resources/model-best'

nlp = spacy.load(spacyModel)
# ner = spacy.load(nerModel)

def extractOrg(text):
    doc = nlp(text)
    # doc = ner(text)
    # custom model use 'BRAND' instead of 'ORG'
    # orgEnts = [ent.text for ent in doc.ents if ent.label_ == 'BRAND' or ent.label_ == 'ORG']
    orgEnts = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
    if len(orgEnts) > 0:
        return orgEnts
    return

def extractProductOllama(text):
    prompt = generatePrompt(text)
    url = 'http://94.100.26.141:25789/api/generate'
    data = {
        'model': 'gemma:2b',
        'prompt': prompt,
        'json': True,
        'stream': False
    }
    response = requests.post(url, json = data)
    return response.json()

def generatePrompt(text):
    prompt = """
        [instruction]: extract brand's name in comman separated format from [tweet] below. if you cannot extract NER just reply with 'no entity'
        [tweet]:
    """
    prompt += text
    return prompt

def extractProduct(text):
    doc = nlp(text)
    productEnts = [ent.text for ent in doc.ents if ent.label_ == 'PRODUCT']
    if len(productEnts) > 0:
        return productEnts[0]
    return

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
    text = text.replace('rT', '')
    
    text = text.replace('#', '')

    tokens = text.split(' ')
    tokens = [token for token in tokens if token.lower() not in stopWords]
    
    cleanedText = ' '.join(tokens).lower().strip()

    return cleanedText