import pandas as pd
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize


# Entrainement du modèle Word2Vec
sentences = X_train['designation'].tolist() + X_train['description'].tolist()  
model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Fonction pour obtenir l'embedding d'un texte en faisant la moyenne des embeddings des mots du texte
def get_text_embedding(text, model_w2v):
    embeddings = [model_w2v.wv[word] for word in text if word in model_w2v.wv]
    if len(embeddings) == 0:
        return [0] * model_w2v.vector_size  
    return sum(embeddings) / len(embeddings) 

# Obtenir les embeddings pour X_train et X_test
X_train_vect_designation = X_train['designation'].apply(lambda x: get_text_embedding(x, model_w2v))
X_test_vect_designation = X_test['designation'].apply(lambda x: get_text_embedding(x, model_w2v))

X_train_vect_description = X_train['description'].apply(lambda x: get_text_embedding(x, model_w2v))
X_test_vect_description = X_test['description'].apply(lambda x: get_text_embedding(x, model_w2v))

# Convertir les séries en matrices
X_train_vect_designation = pd.DataFrame(X_train_vect_designation.tolist())
X_test_vect_designation = pd.DataFrame(X_test_vect_designation.tolist())

X_train_vect_description = pd.DataFrame(X_train_vect_description.tolist())
X_test_vect_description = pd.DataFrame(X_test_vect_description.tolist())

# Dimensions des matrices après vectorisation
print(f"Dimensions de la matrice de vectorisation pour X_train (designation) : {X_train_vect_designation.shape}")
print(f"Dimensions de la matrice de vectorisation pour X_test (designation) : {X_test_vect_designation.shape}")

print(f"Dimensions de la matrice de vectorisation pour X_train (description) : {X_train_vect_description.shape}")
print(f"Dimensions de la matrice de vectorisation pour X_test (description) : {X_test_vect_description.shape}")

# Premières lignes des matrices vectorisées
print("Premières lignes des matrices vectorisées (X_train - designation) :")
print(X_train_vect_designation.head())

print("Premières lignes des matrices vectorisées (X_test - designation) :")
print(X_test_vect_designation.head())

print("Premières lignes des matrices vectorisées (X_train - description) :")
print(X_train_vect_description.head())

print("Premières lignes des matrices vectorisées (X_test - description) :")
print(X_test_vect_description.head())
