import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')

data = pd.read_csv('articles.csv', encoding = 'latin')
data['Article'] = data.Article
data['Title'] = data.Title
data['Article_Title'] = data.Article + " " + data.Title
data.drop_duplicates(inplace = True)
data.reset_index(drop = True, inplace = True)

vectorizer = TfidfVectorizer()
test_matrix = vectorizer.fit_transform(data.Article_Title)
test_matrix = test_matrix.toarray()

nn = NearestNeighbors()
nn.fit(test_matrix)

en_stopwords = stopwords.words("english")
stemmer = PorterStemmer()

def clean(text):
    text = re.sub("[^A-Za-z1-9 ]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    clean_list = []
    for token in tokens:
        if token not in en_stopwords:
            clean_list.append(stemmer.stem(token))
    return " ".join(clean_list)

def Recommender(text):
    text = clean(text)
    text_matrix = vectorizer.transform([text])
    result = nn.kneighbors(n_neighbors=5, X=text_matrix, return_distance=False)
    
    return data.iloc[result[0]].Title.values.tolist()
