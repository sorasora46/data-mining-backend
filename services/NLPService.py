import spacy
import requests
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

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
    SYSTEM_PROMPT = "You are a smart and intelligent Named Entity Recognition (NER) system. I will provide you the definition of the entities you need to extract and the sentence from which you need to extract the entities and the output in given format with examples."
    USER_PROMPT_1 = "Are you clear about your role?"
    ASSISTANT_PROMPT_1 = "Sure, I'm ready to help you with your NER task. Please provide me with the necessary information to get started."
    url = 'http://94.100.26.141:25789/api/chat'
    data = {
        'model': 'gemma:2b',
        'messages': [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_1},
            {"role": "assistant", "content": ASSISTANT_PROMPT_1},
            {"role": "user", "content": prompt}
        ],
        'json': True,
        'stream': False,
        'options': { # for reproducible result
            'seed': 42,
            'temperature': 0
        },
        'keep-alive': '20m'
    }
    response = requests.post(url, json = data)
    return response.json()

def generatePrompt(text):
    PROMPT = (
        "Entity Definition:\n"
        "1. PERSON: Short name or full name of a person from any geographic regions.\n"
        "2. DATE: Any format of dates. Dates can also be in natural language.\n"
        "3. LOC: Name of any geographic location, like cities, countries, continents, districts etc.\n"
        "4. ORG: Name of the companies like Google, samsung, Apple etc.\n"
        "5. NUMBERS: Numerical entites which are numerically present or mentioned in words like 7000, half of dozen etc.\n"
        "\n"
        "Output Format:\n"
        "{{'PERSON': [list of entities present], 'DATE': [list of entities present], 'LOC': [list of entities present],'ORG': [list of entities present],'NUMBERS': [list of entities present]}}\n"
        "If no entities are presented in any categories keep it None\n"
        "\n"
        "Examples:\n"
        "\n"
        "1. Sentence: USA and India are friends. G20 summit going to held in India in September 2023. Indian Prime Minister Narendra Modi will be hosting it and TATA will be giving charity of $150 Million.\n"
        "Output: {{'PERSON': ['Narendra Modi'], 'DATE': ['September 2023'], 'LOC': ['USA','India','India'],'ORG':['TATA'],'NUMBERS':['150 Million']}}\n"
        "\n"
        "2. Sentence: Mr.John and Sunita Roy are friends and they meet each other on 24/03/1998 in Samsung while they were co-workers and shared Rs.8000 in exchange for some work.\n"
        "Output: {{'PERSON': ['Mr. John', 'Sunita Roy'], 'DATE': ['24/03/1998'], 'LOC': ['None'],'ORG':['Samsung'],'NUMBERS':['8000']}}\n"
        "\n"
        "Input:\n"
        "Sentence: {}\n"
        "Output: "
    )
    prompt = PROMPT.format(text)
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