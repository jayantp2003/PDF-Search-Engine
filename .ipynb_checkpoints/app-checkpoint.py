import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
import math
import nltk
import tempfile


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stopword = set(stopwords.words('english'))

@st.cache_data
def preprocess(content):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    twords = word_tokenize(content.lower())
    twords = [lemmatizer.lemmatize(word) for word in twords if word.isalpha() and word not in stopword]
    word_freq = Counter(twords)
    return dict(word_freq)

@st.cache_data
def pagerank(query, wordfreqs):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    twords = word_tokenize(query.lower())
    twords = [lemmatizer.lemmatize(word) for word in twords if word.isalpha() and word not in stopword]
    
    sum1 = {word: 0 for word in twords}
    pagefreq = {word: 0.0001 for word in twords}
    
    for word in twords:
        for i in wordfreqs:
            if word in wordfreqs[i].keys():
                sum1[word] += wordfreqs[i][word]
                pagefreq[word] += 1

    tf = {}
    for word in twords:
        tf[word] = {}
        for i in wordfreqs:
            if word in wordfreqs[i].keys():
                tf[word][i] = (wordfreqs[i][word] / sum1[word])
            else:
                tf[word][i] = 0

    idf = {}
    totalpages = len(wordfreqs.keys())
    for word in twords:
        idf[word] = math.log(totalpages / pagefreq[word])

    tfidfscore = {}
    pagerankscore = {i: 0 for i in range(totalpages)}
    
    for word in twords:
        tfidfscore[word] = {}
        for i in range(totalpages):
            tfidfscore[word][i] = tf[word][i] * idf[word]
            pagerankscore[i] += tfidfscore[word][i]

    return pagerankscore

def main():
    st.title("PDF Page Rank Application")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name
        loader = PyPDFLoader(file_path=temp_file_path, extract_images=True)
        docs = loader.lazy_load()
        docs1 = [doc for doc in docs]
        wordfreqs = {doc.metadata['page']: preprocess(doc.page_content) for doc in docs1}
        query = st.text_input("Enter your query:")
        
        if query:
            ranks = pagerank(query, wordfreqs)
            scores = ranks
            sorted_indices = sorted(scores, key=scores.get, reverse=True)
            rankings = {index: rank + 1 for rank, index in enumerate(sorted_indices)}

            st.write("Ranked Page Numbers:")
            for page_num, rank in rankings.items():
                st.write(f"Page {page_num}: Rank {rank}")
                if st.button(f"View Page {page_num}"):
                    st.write(docs1[page_num].page_content)

if __name__ == "__main__":
    main()