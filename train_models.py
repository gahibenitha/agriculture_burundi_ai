import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# CHARGEMENT DU DATASET
# =========================

df = pd.read_csv("agriculture_burundi.csv")

print("Dataset chargé avec succès")
print(df.head())

# =========================
# VALEURS MANQUANTES
# =========================

print("\nValeurs manquantes :")
print(df.isnull().sum())

# Remplissage des colonnes numériques par la médiane
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_columns:
    df[col].fillna(df[col].median(), inplace=True)

# Remplissage des colonnes catégorielles par le mode
categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nValeurs manquantes après traitement :")
print(df.isnull().sum())

# =========================
# ENCODAGE DES VARIABLES
# =========================

df = pd.get_dummies(df, columns=[
    'province',
    'culture',
    'saison'
], drop_first=True)

# =========================
# X ET y
# =========================

X = df.drop([
    'bonne_recolte',
    'rendement_t_ha',
    'production_totale_t'
], axis=1)

y = df['bonne_recolte']

# =========================
# SAUVEGARDE DES COLONNES
# =========================

joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

# =========================
# NORMALISATION
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Sauvegarde du scaler
joblib.dump(scaler, "models/scaler.pkl")

# =========================
# TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# DECISION TREE
# =========================

decision_tree = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

decision_tree.fit(X_train, y_train)

y_pred_dt = decision_tree.predict(X_test)

acc_dt = accuracy_score(y_test, y_pred_dt)

print("\nAccuracy Decision Tree :", acc_dt)

joblib.dump(decision_tree, "models/decision_tree.pkl")

# =========================
# RANDOM FOREST
# =========================

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X_train, y_train)

y_pred_rf = random_forest.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)

print("Accuracy Random Forest :", acc_rf)

joblib.dump(random_forest, "models/random_forest.pkl")

# =========================
# LOGISTIC REGRESSION
# =========================

logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_model.fit(X_train, y_train)

y_pred_lr = logistic_model.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)

print("Accuracy Logistic Regression :", acc_lr)

joblib.dump(logistic_model, "models/logistic_regression.pkl")

# =========================
# FIN
# =========================

print("\nTous les modèles ont été entraînés et sauvegardés.")