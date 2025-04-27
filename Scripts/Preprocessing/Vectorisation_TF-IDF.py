import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialiser les vectoriseurs TF-IDF pour 'designation' et 'description'
vectorize_layer_designation = TfidfVectorizer(max_features=5000, stop_words='french')  # Vous pouvez ajuster max_features selon vos besoins
vectorize_layer_description = TfidfVectorizer(max_features=5000, stop_words='french')

# Application du TF-IDF
X_train_vect_designation = vectorize_layer_designation.fit_transform(X_train['designation'].astype('str'))
X_test_vect_designation = vectorize_layer_designation.transform(X_test['designation'].astype('str'))

X_train_vect_description = vectorize_layer_description.fit_transform(X_train['description'].astype('str'))
X_test_vect_description = vectorize_layer_description.transform(X_test['description'].astype('str'))

# Dimensions des matrices après vectorisation
print(f"Dimensions de la matrice de vectorisation pour X_train (designation) : {X_train_vect_designation.shape}")
print(f"Dimensions de la matrice de vectorisation pour X_test (designation) : {X_test_vect_designation.shape}")

print(f"Dimensions de la matrice de vectorisation pour X_train (description) : {X_train_vect_description.shape}")
print(f"Dimensions de la matrice de vectorisation pour X_test (description) : {X_test_vect_description.shape}")

# "Premières lignes des matrices vectorisées
print("Premières lignes des matrices vectorisées (X_train - designation) :")
print(X_train_vect_designation[:5].toarray())  

print("Premières lignes des matrices vectorisées (X_test - designation) :")
print(X_test_vect_designation[:5].toarray())

print("Premières lignes des matrices vectorisées (X_train - description) :")
print(X_train_vect_description[:5].toarray())

print("Premières lignes des matrices vectorisées (X_test - description) :")
print(X_test_vect_description[:5].toarray())
