# PDF Search Engine

This project is a PDF search engine that utilizes Natural Language Processing (NLP) techniques to extract and rank pages based on user queries. The application analyzes the content of PDF documents and provides real-time ranking for pages relevant to the user's input.

## Features

- **PDF Upload**: Users can upload PDF files for analysis.
- **Real-Time Query Processing**: The search engine provides real-time ranking of PDF pages based on user queries.
- **NLP Techniques**: Utilizes tokenization, stemming, lemmatization, and TF-IDF for improved search accuracy.
- **Page Ranking**: Returns ranked pages based on relevance to the query, allowing users to directly view content from specific pages.

## Technologies Used

- **Python**: The main programming language for development.
- **Streamlit**: A framework to create the web application.
- **NLTK**: Natural Language Toolkit for NLP tasks.
- **LangChain**: For handling PDF document loading.

## Requirements

Make sure you have the following libraries installed:

```bash
pip install streamlit langchain_community nltk
```
You also need to download the necessary NLTK resources by running:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/pdf-search-engine.git
   cd pdf-search-engine
   ```
2. **Run the Streamlit application:**

   ```bash
   streamlit run app.py
   ```
3. Upload a PDF document and enter your query in the provided text input. The application will display ranked pages based on the relevance of the query.

## How It Works

 1. **Preprocessing:** The application preprocesses the PDF content by tokenizing, stemming, lemmatizing, and removing stopwords to generate a word frequency dictionary for each page.
   
 2. **PageRank Algorithm:** The application implements a PageRank-like algorithm that calculates TF-IDF scores for words in the query and ranks the pages accordingly

 3. **Results:** Users can view the ranked pages and click on the page number to see the content directly.

Thank you!!






