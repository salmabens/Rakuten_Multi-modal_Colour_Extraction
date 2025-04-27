import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '../Data') 
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../Output')  

X_train = pd.read_csv(os.path.join(DATA_PATH, "X_train_update.csv"))
X_test = pd.read_csv(os.path.join(DATA_PATH, "X_test_update.csv"))
y_train = pd.read_csv(os.path.join(DATA_PATH, "Y_train_CVw08PX.csv"))
y_train_labels = y_train['prdtypecode'] 

# Réindexer les labels
label_encoder = LabelEncoder()
y_train_labels = label_encoder.fit_transform(y_train_labels)  

print("Classes uniques après réindexation :")
print(label_encoder.classes_)

max_sequence_length = 100
num_classes = len(np.unique(y_train_labels))
embedding_dim = 50

X_train['designation'] = X_train['designation'].apply(preprocess_text)
X_test['designation'] = X_test['designation'].apply(preprocess_text)
X_train['description'] = X_train['description'].apply(preprocess_text)
X_test['description'] = X_test['description'].apply(preprocess_text)

# Utilisation de TfidfVectorizer pour 'designation' et 'description'
vectorize_layer_designation = TfidfVectorizer(max_features=5000, stop_words='french')
vectorize_layer_description = TfidfVectorizer(max_features=5000, stop_words='french')

# Appliquer TF-IDF sur la colonne 'designation'
X_train_vect_designation = vectorize_layer_designation.fit_transform(X_train['designation'].astype('str'))
X_test_vect_designation = vectorize_layer_designation.transform(X_test['designation'].astype('str'))

# Appliquer TF-IDF sur la colonne 'description'
X_train_vect_description = vectorize_layer_description.fit_transform(X_train['description'].astype('str'))
X_test_vect_description = vectorize_layer_description.transform(X_test['description'].astype('str'))

# Combiner les vecteurs pour 'designation' et 'description'
X_train_combined = np.concatenate([X_train_vect_designation.toarray(), X_train_vect_description.toarray()], axis=1)
X_test_combined = np.concatenate([X_test_vect_designation.toarray(), X_test_vect_description.toarray()], axis=1)

# Padding des séquences pour qu'elles aient toutes la même longueur
X_train_padded = pad_sequences(X_train_combined, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_combined, maxlen=max_sequence_length)

# Construction du modèle CNN
model = Sequential()
model.add(Embedding(input_dim=vectorize_layer_designation.vocabulary_.size(), output_dim=embedding_dim, input_length=max_sequence_length))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()

history = model.fit(X_train_padded, y_train_labels, epochs=10, batch_size=32, validation_split=0.2, callbacks=[TqdmCallback(verbose=2)])

model.save(os.path.join(DATA_OUT, "choisi-un-nom"))

y_pred = np.argmax(model.predict(X_test_padded), axis=1)

print("Scores de chaque époque :")
for epoch in range(10):
    print(f"Epoch {epoch + 1}:")
    print(f"  - Loss: {history.history['loss'][epoch]:.4f}")
    print(f"  - Accuracy: {history.history['accuracy'][epoch]:.4f}")
    print(f"  - Val_loss: {history.history['val_loss'][epoch]:.4f}")
    print(f"  - Val_accuracy: {history.history['val_accuracy'][epoch]:.4f}")