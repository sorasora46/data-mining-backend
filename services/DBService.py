import pandas as pd

dataset = pd.read_csv('resources/tweets-engagement-metrics.csv')

def searchTweetByOrg(text, n = 20):
    targetDataFrame = dataset[dataset['text'].str.contains(text, case=False)]
    firstNRows = targetDataFrame.head(n)
    results = firstNRows['text'].tolist()
    resultsNoDuplicate = set(results)
    return list(resultsNoDuplicate)

def searchTweetByMultipleOrgs(orgs, n = 20):
    result = listToDictEmptyArray(orgs)
    for org in orgs:
        result[org] = searchTweetByOrg(org, n)
    return result
    
def listToDictEmptyArray(list):
    dict = {}
    for i in range(len(list)):
        dict[list[i]] = []
    return dict
