# 🌾 Agriculture Burundi AI

## Prédiction Intelligente des Récoltes Agricoles au Burundi

Application web développée avec **Python**, **Machine Learning** et **Streamlit** permettant de prédire la qualité des récoltes agricoles au Burundi à partir de données climatiques, géographiques et agricoles.

---

# 📖 Description du projet

L'agriculture constitue l'un des secteurs les plus importants de l'économie burundaise.

Ce projet utilise plusieurs algorithmes de Machine Learning afin de prédire si une récolte sera :

* ✅ Bonne récolte
* ❌ Mauvaise récolte

à partir de plusieurs facteurs tels que :

* La province
* La culture cultivée
* La saison agricole
* L'altitude
* La pluviométrie
* La température
* L'utilisation d'engrais
* L'irrigation
* La superficie cultivée

L'objectif est d'aider les agriculteurs et les décideurs à mieux anticiper les résultats agricoles.

---

# 🎯 Objectifs

* Analyser les données agricoles du Burundi
* Nettoyer et préparer les données
* Construire plusieurs modèles prédictifs
* Comparer les performances des modèles
* Développer une application web interactive
* Fournir des recommandations agricoles

---

# 🛠️ Technologies utilisées

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Streamlit
* Joblib
* Git & GitHub

---

# 🤖 Modèles de Machine Learning

Les modèles utilisés dans ce projet sont :

### 1. Decision Tree Classifier

Modèle basé sur un arbre de décision permettant d'expliquer facilement les prédictions.

### 2. Random Forest Classifier

Ensemble de plusieurs arbres de décision offrant généralement de meilleures performances.

### 3. Logistic Regression

Modèle statistique utilisé pour la classification binaire.

---

# 📂 Structure du projet

```text
TP_AGRICULTURE_BURUNDI/
│
├── app/
│   └── app.py
│
├── images/
│
├── models/
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── agriculture_burundi.csv
├── README.md
├── requirements.txt
├── tp_agriculture.ipynb
└── train_models.py
```

---

# 📊 Jeu de données

Le dataset contient plusieurs informations agricoles :

| Variable            | Description                 |
| ------------------- | --------------------------- |
| annee               | Année agricole              |
| saison              | Saison agricole             |
| province            | Province du Burundi         |
| culture             | Type de culture             |
| altitude_m          | Altitude en mètres          |
| pluviometrie_mm     | Quantité de pluie           |
| temperature_moy_C   | Température moyenne         |
| superficie_ha       | Surface cultivée            |
| utilisation_engrais | Utilisation d'engrais       |
| acces_irrigation    | Accès à l'irrigation        |
| nb_menages          | Nombre de ménages agricoles |
| bonne_recolte       | Variable cible              |

---

# ⚙️ Installation

## 1. Cloner le projet

```bash
git clone https://github.com/VOTRE-NOM/agriculture_burundi_ai.git
cd agriculture_burundi_ai
```

## 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

# 📈 Entraîner les modèles

Exécuter :

```bash
python train_models.py
```

Cette commande génère automatiquement :

```text
decision_tree.pkl
random_forest.pkl
logistic_regression.pkl
scaler.pkl
feature_columns.pkl
```

dans le dossier :

```text
models/
```

---

# 🚀 Lancer l'application

```bash
streamlit run app/app.py
```

ou

```bash
python -m streamlit run app/app.py
```

---

# 🌐 Fonctionnalités de l'application

L'application permet :

* Sélectionner une province
* Sélectionner une culture
* Choisir une saison agricole
* Saisir les données climatiques
* Choisir un modèle IA
* Obtenir une prédiction
* Visualiser la probabilité de succès
* Recevoir des recommandations agricoles

---

# 📋 Exemple de résultat

```text
Bonne Récolte
Probabilité : 92%
```

ou

```text
Mauvaise Récolte
Probabilité : 38%
```

---

# 📊 Visualisations

L'application affiche également :

* Graphiques de probabilité
* Comparaison des résultats
* Statistiques descriptives
* Analyse exploratoire des données


# 👨‍💻 Auteur

## GAHIMBARE Benitha
Projet académique réalisé dans le cadre de l'application du Machine Learning à l'agriculture au Burundi.
---


