import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# =========================
# CONFIGURATION
# =========================
st.set_page_config(
    page_title="🌾 Récoltes Burundi",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CHARGEMENT MODÈLES
# =========================
@st.cache_resource
def load_models():

    base = os.path.join(os.path.dirname(__file__), '..', 'models')

    fichiers = {
        'arbre_decision': 'decision_tree.pkl',
        'foret_aleatoire': 'random_forest.pkl',
        'regression_logistique': 'logistic_regression.pkl',
        'scaler': 'scaler.pkl',
        'colonnes': 'feature_columns.pkl'
    }

    models = {}

    for nom, fichier in fichiers.items():

        chemin = os.path.join(base, fichier)

        if not os.path.exists(chemin):
            st.error(f"❌ Fichier introuvable : {fichier}")
            st.stop()

        if os.path.getsize(chemin) == 0:
            st.error(f"❌ Fichier vide : {fichier}")
            st.stop()

        try:
            models[nom] = joblib.load(chemin)

        except Exception as e:
            st.error(f"❌ Erreur chargement : {fichier}")
            st.code(str(e))
            st.stop()

    return models


data = load_models()

# =========================
# SIDEBAR
# =========================
st.sidebar.image("https://flagcdn.com/w80/bi.png", width=60)
st.sidebar.title("⚙️ Configuration")

modele_nom = st.sidebar.selectbox(
    "🤖 Modèle à utiliser",
    ["Forêt Aléatoire", "Arbre de Décision", "Régression Logistique"]
)

modele_map = {
    "Arbre de Décision": "arbre_decision",
    "Forêt Aléatoire": "foret_aleatoire",
    "Régression Logistique": "regression_logistique"
}

modele = data[modele_map[modele_nom]]

# =========================
# TITRE
# =========================
st.title("🌾 Prédiction des Récoltes au Burundi")
st.markdown("Prédisez si la récolte sera bonne ou mauvaise selon les conditions agricoles.")

st.divider()

# =========================
# FORMULAIRE
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    province = st.selectbox("Province", [
        "Bubanza","Bujumbura Rural","Bururi","Cankuzo","Cibitoke",
        "Gitega","Karuzi","Kayanza","Kirundo","Makamba",
        "Muramvya","Muyinga","Mwaro","Ngozi","Rutana"
    ])

    culture = st.selectbox("Culture", [
        "Maïs","Haricot","Manioc","Patate douce","Sorgho","Bananier"
    ])

    saison = st.selectbox("Saison", ["A (Mars–Juin)", "B (Sept–Déc)"])
    saison_code = "A" if "A" in saison else "B"

with col2:
    altitude = st.slider("Altitude (m)", 500, 2800, 1500)
    pluviometrie = st.slider("Pluviométrie (mm)", 100, 1500, 800)
    temperature = st.slider("Température (°C)", 14.0, 28.0, 20.0)

with col3:
    superficie = st.slider("Superficie (ha)", 0.5, 20.0, 3.0)
    nb_menages = st.slider("Nombre ménages", 10, 500, 100)
    engrais = st.radio("Engrais", ["Oui", "Non"])
    irrigation = st.radio("Irrigation", ["Oui", "Non"])

st.divider()

# =========================
# PRÉDICTION
# =========================
if st.button("🔍 Prédire", use_container_width=True):

    input_data = {
        'annee': 2024,
        'altitude_m': altitude,
        'pluviometrie_mm': pluviometrie,
        'temperature_moy_C': temperature,
        'superficie_ha': superficie,
        'utilisation_engrais': 1 if engrais == "Oui" else 0,
        'acces_irrigation': 1 if irrigation == "Oui" else 0,
        'nb_menages': nb_menages,
    }

    df_input = pd.DataFrame([input_data])

    # encodage catégoriel
    df_input[f'saison_{saison_code}'] = 1
    df_input[f'province_{province}'] = 1
    df_input[f'culture_{culture}'] = 1

    # alignement colonnes
    for col in data['colonnes']:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[data['colonnes']]

    # scaling
    X_scaled = data['scaler'].transform(df_input)

    # prediction
    prediction = modele.predict(X_scaled)[0]
    proba = modele.predict_proba(X_scaled)[0]

    st.subheader("📊 Résultat")

    if prediction == 1:
        st.success("✅ BONNE RÉCOLTE")
        prob = proba[1]
    else:
        st.error("❌ MAUVAISE RÉCOLTE")
        prob = proba[0]

    st.metric("Probabilité", f"{prob*100:.2f}%")
    st.metric("Modèle", modele_nom)

    # graphique
    fig, ax = plt.subplots()
    ax.bar(["Mauvaise", "Bonne"], proba)
    st.pyplot(fig)

    # conseil
    st.subheader("💡 Conseil")

    if prediction == 1 and prob > 0.75:
        st.success("Très bonnes conditions agricoles 🌱")
    elif pluviometrie < 500:
        st.warning("Faible pluie, envisager irrigation 💧")
    elif engrais == "Non":
        st.warning("Utiliser des engrais recommandé 🌿")
    else:
        st.info("Conditions moyennes, surveillance recommandée ⚠️")