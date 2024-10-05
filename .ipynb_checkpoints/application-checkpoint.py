from langchain_community.document_loaders import PyPDFLoader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
import numpy as np
import nltk
import math
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

stopword = set(stopwords.words('english'))
loader = PyPDFLoader(file_path='Report.pdf',extract_images=True)
docs = loader.lazy_load()
docs1=[]
for doc in docs:
    docs1.append(doc)

def preprocess(content):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    twords = word_tokenize(text=content.lower())
    twords = [lemmatizer.lemmatize(word) for word in twords if word.isalpha() and word not in stopword]
    word_freq = Counter(twords)
    word_freq_dict = dict(word_freq)
    return word_freq_dict


wordfreqs={}
for i in docs1:
    wordfreqs[i.metadata['page']]=preprocess(i.page_content)


def pagerank(query):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    twords = word_tokenize(text=query.lower())
    twords = [lemmatizer.lemmatize(word) for word in twords if word.isalpha() and word not in stopword]
    sum1 = {word:0 for word in twords}
    pagefreq = {word:0.0001 for word in twords}
    for word in twords:
        for i in wordfreqs:
            if word in wordfreqs[i].keys():
                sum1[word]+=wordfreqs[i][word]
                pagefreq[word]+=1
    tf = {}
    for word in twords:
        tf[word] = {}          
        for i in wordfreqs:
            if word in wordfreqs[i].keys():
                tf[word][i]=(wordfreqs[i][word]/sum1[word])
            else:
                tf[word][i]=0


    idf = {}
    totalpages = len(wordfreqs.keys())
    for word in twords:
        idf[word] = math.log(totalpages/pagefreq[word])

    tfidfscore = {}
    pagerankscore = {i:0 for i in range(totalpages)}
    for word in twords:
        tfidfscore[word]={}
        for i in range(totalpages):
             tfidfscore[word][i]=tf[word][i]*idf[word]
             pagerankscore[i]+=tfidfscore[word][i]
    return pagerankscore

ranks = pagerank("oracle internship automation spring boot framework is am are was were has")

scores = ranks
sorted_indices = sorted(scores, key=scores.get, reverse=True)
rankings = {index: rank + 1 for rank, index in enumerate(sorted_indices)}
print(rankings)