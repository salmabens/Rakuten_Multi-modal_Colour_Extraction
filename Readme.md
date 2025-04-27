# Rakuten_Multi-modal_Colour_Extraction

## Description

Ce projet est basé sur le challenge organisé par le **Rakuten Institute of Technology (RIT)**, visant à prédire les **codes de type de produit** à partir de données **multimodales** (images et textes) du catalogue de **Rakuten Ichiba**. L’objectif est de prédire le **code de type de produit** pour chaque article, en utilisant une approche combinée des informations des **images**, **titres** et **descriptions**.

Les données sont particulièrement difficiles à traiter en raison de plusieurs facteurs :

- Des descriptions mal rédigées et des **données manquantes** ou **bruitées**.
- Un **déséquilibre des classes**, où certaines catégories de produits sont plus fréquentes que d'autres.

La tâche implique la gestion de **250 000 produits**, avec des données potentiellement bruitées et un déséquilibre des classes. Le défi réside dans l'utilisation de modèles de **classification multi-classe**, où chaque produit est associé à un **code de type unique**.

L'approche multimodale combine des techniques de **computer vision** et de **traitement du langage naturel (NLP)** pour améliorer la précision de la classification, en traitant des **images** et des **descriptions** de produits dans un environnement à grande échelle.


## 🛠 Prérequis

Pour utiliser ce projet, vous devez d'abord créer un environnement virtuel et installer les packages répertoriés dans le fichier `requirements.txt`.

## Installation

1. Cloner le dépôt sur votre machine :
   ```bash
   git clone https://github.com/salmabens/Rakuten_Multi-modal_Colour_Extraction.git
   cd rakuten_Multi-modal_Colour_Extraction

2. Installer les dépendances avec pip
   ```bash
   pip install -r requirements.txt

## 📁 Structure du projet
``` 
/Scripts          # Contient tous les scripts du projet
    ├── Modèle (image)  # Scripts liés aux modèles utilisant uniquement des images
    ├── Modèle (texte + image)  # Scripts pour les modèles texte + image
    ├── Modèle (texte)  # Scripts pour les modèles utilisant uniquement du texte
    ├── Preprocessing    # Scripts de prétraitement des données
    └── EDA.ipynb         # Notebook principal pour l'analyse exploratoire des données (EDA)

/Output           # Résultats des modèles et autres sorties
    └── [Résultats générés ici]
    
Presentation.pdf
Rapport.pdf
README.md          # Documentation du projet
Requirements.txt   # Liste des dépendances Python nécessaires pour exécuter le projet
```
## 👥 Contributeurs

- **Salma BENMOUSSA**
- **Charlotte CEGARRA**

Ce projet a été développé dans le cadre du Master MOSEF, à l'université Paris 1 Panthéon Sorbonne.

## 📩 Contact

N'hésitez pas à nous contacter pour toute question :

- salmabenmoussa103@gmail.com 
- charlottecegarrapro@gmail.com
