import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from tqdm.keras import TqdmCallback  
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

# Paramètres
max_sequence_length = 100  
num_classes = len(np.unique(y_train_labels))  

# Prétraiter les colonnes 'designation' et 'description' (assurez-vous qu'elles sont nettoyées et prétraitées)
X_train['designation'] = X_train['designation'].apply(preprocess_text)
X_test['designation'] = X_test['designation'].apply(preprocess_text)

X_train['description'] = X_train['description'].apply(preprocess_text)
X_test['description'] = X_test['description'].apply(preprocess_text)

# Utilisation du tokenizer BERT pour convertir le texte en tokens que BERT peut comprendre
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_text(text):
    return tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=max_sequence_length, return_tensors='tf')

# Appliquer le tokenizer aux colonnes 'designation' et 'description'
X_train_encoded_designation = [encode_text(text) for text in X_train['designation']]
X_test_encoded_designation = [encode_text(text) for text in X_test['designation']]

X_train_encoded_description = [encode_text(text) for text in X_train['description']]
X_test_encoded_description = [encode_text(text) for text in X_test['description']]

# Extraire les input_ids et les attention_masks pour les passer au modèle BERT
train_input_ids_designation = np.array([encoded['input_ids'] for encoded in X_train_encoded_designation])
test_input_ids_designation = np.array([encoded['input_ids'] for encoded in X_test_encoded_designation])

train_attention_mask_designation = np.array([encoded['attention_mask'] for encoded in X_train_encoded_designation])
test_attention_mask_designation = np.array([encoded['attention_mask'] for encoded in X_test_encoded_designation])

train_input_ids_description = np.array([encoded['input_ids'] for encoded in X_train_encoded_description])
test_input_ids_description = np.array([encoded['input_ids'] for encoded in X_test_encoded_description])

train_attention_mask_description = np.array([encoded['attention_mask'] for encoded in X_train_encoded_description])
test_attention_mask_description = np.array([encoded['attention_mask'] for encoded in X_test_encoded_description])

# Charger le modèle BERT
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Construire le modèle avec les embeddings de BERT
input_ids = tf.keras.layers.Input(shape=(max_sequence_length,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.layers.Input(shape=(max_sequence_length,), dtype=tf.int32, name="attention_mask")

# Passer les inputs à BERT
bert_output = bert_model(input_ids, attention_mask=attention_mask)

# Prendre la sortie du CLS token pour la classification
cls_output = bert_output.last_hidden_state[:, 0, :]

# Ajouter des couches supplémentaires pour la classification
x = tf.keras.layers.Dense(128, activation='relu')(cls_output)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

# Créer le modèle
model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

# Compiler le modèle
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Entraînement du modèle
history = model.fit(
    [train_input_ids_designation, train_attention_mask_designation], y_train_labels, 
    epochs=10, 
    batch_size=32, 
    validation_split=0.2,
    callbacks=[TqdmCallback(verbose=2)]  
)

# Faire des prédictions sur X_test
y_pred = np.argmax(model.predict([test_input_ids_designation, test_attention_mask_designation]), axis=1)

# Afficher les scores de chaque epoch
print("Scores de chaque époque :")
for epoch in range(10):
    print(f"Epoch {epoch + 1}:")
    print(f"  - Loss: {history.history['loss'][epoch]:.4f}")
    print(f"  - Accuracy: {history.history['accuracy'][epoch]:.4f}")
    print(f"  - Val_loss: {history.history['val_loss'][epoch]:.4f}")
    print(f"  - Val_accuracy: {history.history['val_accuracy'][epoch]:.4f}")

# Calculer le F1-score sur les prédictions
f1 = f1_score(y_train_labels, y_pred, average='weighted')
print(f"F1-score sur les prédictions : {f1:.4f}")
