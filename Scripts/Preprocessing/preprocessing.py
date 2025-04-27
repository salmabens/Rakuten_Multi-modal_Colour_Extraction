import os
import pandas as pd
import nltk
from bs4 import BeautifulSoup
import html
import re
from unidecode import unidecode  
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer

DATA_PATH = os.path.join(os.path.dirname(__file__), '../Data') 
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../Output')  

X_train = pd.read_csv(os.path.join(DATA_PATH, "X_train_update.csv"))
X_test = pd.read_csv(os.path.join(DATA_PATH, "X_test_update.csv"))
y_train = pd.read_csv(os.path.join(DATA_PATH, "Y_train_CVw08PX.csv"))


# Fonction de prétraitement
def preprocess_text(text):
    if not isinstance(text, str):  
        return ''  
    
    # 1. Suppression des balises HTML
    text = BeautifulSoup(text, "html.parser").get_text()

    # 2. Conversion des entités HTML (ex: &#39; -> ')
    text = html.unescape(text)

    # 3. Conversion en minuscules
    text = text.lower()

    # 4. Suppression des accents (ex: é -> e)
    text = unidecode(text)

    # 5. Suppression des caractères spéciaux 
    text = re.sub(r'[^a-z\s]', '', text)  

    # 6. Tokenisation : Diviser le texte en mots
    tokens = word_tokenize(text)

    # 7. Suppression des mots vides (stopwords)
    stop_words = set(stopwords.words('french'))  
    tokens = [word for word in tokens if word not in stop_words]

    # 8. Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # 9. Suppression des mots trop courts et trop fréquents
    freq_dist = FreqDist(tokens)
    tokens = [word for word in tokens if len(word) > 2 and freq_dist[word] < 50]  # Suppression des mots très courts et très fréquents

    return ' '.join(tokens)

# Appliquer le prétraitement sur les colonnes 'designation' et 'description'
X_train['designation'] = X_train['designation'].apply(preprocess_text)
X_test['designation'] = X_test['designation'].apply(preprocess_text)

X_train['description'] = X_train['description'].apply(preprocess_text)
X_test['description'] = X_test['description'].apply(preprocess_text)