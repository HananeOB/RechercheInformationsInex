import re 
import nltk 
from nltk.corpus import stopwords

## Extraire les documents et leurs IDs 
def get_all_docs(textfile="Collection_Texte" ):# Gets all documents as rows of a list
    file = open(textfile)
    text = file.read()
    file.close()
    text = re.sub(r'\n', ' ', text)
    patternN = r'<doc><docno>(\d+)</docno>'
    patternD = r'<doc><docno>\d+</docno>(.*?)</doc>'
    DocN = re.findall(patternN, text) ## list of DocIds 
    DocD = re.findall(patternD, text) ## list of DocTexts

    return [DocN, DocD]

## Creation des Bags of words Pour un DOC
def get_bag_of_words(DocD) :
    DocD = DocD.strip().lower()
    DocD = re.sub(r'\n+','',DocD)
    DocD = re.sub(r',|:|\.|;|\d|\(|\)|\'','',DocD)
    DocD = re.sub(r'-',' ', DocD)
    DocD = re.sub(r'\s+',' ', DocD)
    bag_words = DocD.split(" ")
    bag_words = list(filter(lambda x : x not in stopwords.words('english'), bag_words ))
    bag_words = list(filter(lambda x : len(x)>1, bag_words ))
    return(bag_words)

## creation des 


docIDs,docTexts = get_all_docs() 
# Bag = set()
# for DocD in docTexts :
#     Bag = Bag.union(get_bag_of_words(DocD))

L = []
for docD in docTexts :
    numOfWords = dict.fromkeys(Bag, 0)
    bagOfWords = get_bag_of_words(DocD)
    for word in bagOfWords:
        numOfWords[word] += 1
    L.append(numOfWords)


        


# nltk.download('stopwords')
# L = nltk.corpus.stopwords.words('english')
# print(L)

