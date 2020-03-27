import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('punkt') # if necessary...
# nltk.download('stopwords')


def stem_tokens(tokens):
    stemmer = nltk.stem.porter.PorterStemmer()
    return [stemmer.stem(item) for item in tokens]


def normalize(text):
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))


def cosine_sim(text1: string, text2: string) -> float:
    """

    :rtype: object
    """
    vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')
    tfidf = vectorizer.fit_transform([text1, text2])
    return (tfidf * tfidf.T).A[0, 1]
